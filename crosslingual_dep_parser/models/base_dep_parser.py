from typing import Dict, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, InputVariationalDropout, Embedding
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import get_device_of, masked_log_softmax, get_range_vector
from allennlp.nn.util import (
    get_lengths_from_binary_sequence_mask,
)
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores

logger = logging.getLogger(__name__)


@Model.register("BaseDependencyParser")
class BaseDependencyParser(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        postags_embedding: Embedding,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        arc_representation_dim: int,
        head_arc_feedforward: FeedForward = None,
        child_arc_feedforward: FeedForward = None,
        head_tag_feedforward: FeedForward = None,
        child_tag_feedforward: FeedForward = None,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.postags_embedding = postags_embedding
        self.lstm_encoder = encoder

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = head_arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("relu")()
        )
        self.child_arc_feedforward = child_arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("relu")()
        )

        self.head_tag_feedforward = head_tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("relu")()
        )
        self.child_tag_feedforward = child_tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("relu")()
        )

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        self.num_labels = self.vocab.get_vocab_size("deprels_tags")
        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, self.num_labels
        )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()+postags_embedding.get_output_dim()

        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self.mst_decoding = use_mst_decoding_for_validation

        self._attachment_scores = AttachmentScores()
        initializer(self)

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        postags: torch.LongTensor,
        language: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        heads: torch.LongTensor = None,
        deprels: torch.LongTensor = None,
        confidences: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        postags, heads, deprels, confidences = self.add_root_index(tokens, postags, heads, deprels, confidences)
        mask = tokens['tokens']['mask']
        embedded_text_input = self.text_field_embedder(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)

        postags_emb = self.postags_embedding(postags)
        postags_emb = self._input_dropout(postags_emb)

        embedded_text_input = torch.cat([
            embedded_text_input, postags_emb
        ], dim=-1)

        encoded_text = self.lstm_encoder(embedded_text_input, mask)
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))

        predicted_heads, predicted_deprels, heads_logits, predicted_confidences = \
            self._parse(
                head_arc_representation, child_arc_representation,
                head_tag_representation, child_tag_representation, mask
            )

        if heads is not None and deprels is not None:
            eva_mask = self._get_mask_for_eval(mask, confidences)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_deprels[:, 1:],
                heads[:, 1:],
                deprels[:, 1:],
                eva_mask[:, 1:],
            )

            arc_nll, tag_nll = self._construct_loss(
                heads_logits, heads,
                head_tag_representation, child_tag_representation, deprels,
                mask, confidences
            )
            loss = arc_nll + tag_nll
        else:
            arc_nll, tag_nll = torch.tensor(0.), torch.tensor(0.)
            loss = arc_nll + tag_nll

        output_dict = {
            "heads": predicted_heads,
            "deprels": predicted_deprels,
            "confidences": predicted_confidences,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
            "metadata": metadata,
        }

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        confidences = output_dict.pop("confidences").cpu().detach().numpy()
        deprels = output_dict.pop("deprels").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        metadata = output_dict.pop("metadata")
        lengths = get_lengths_from_binary_sequence_mask(mask)

        confidences_readable = []
        deprels_readable = []
        heads_readable = []
        for instance_confi, instance_heads, instance_tags, length in zip(confidences, heads, deprels, lengths):
            instance_confi = list(numpy.around(instance_confi[1:length], decimals=2))
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [
                self.vocab.get_token_from_index(label, "deprels_tags") for label in instance_tags
            ]
            confidences_readable.append(instance_confi)
            deprels_readable.append(labels)
            heads_readable.append(instance_heads)

        output_dict["confidences"] = confidences_readable
        output_dict["deprels"] = deprels_readable
        output_dict["heads"] = heads_readable
        output_dict["tokens"] = [meta["tokens"] for meta in metadata]
        output_dict["postags"] = [meta["postags"] for meta in metadata]
        output_dict["language"] = [meta["language"] for meta in metadata]

        output_dict.pop('arc_loss')
        output_dict.pop('tag_loss')
        output_dict.pop('loss')

        return output_dict

    def _parse(
        self,
        head_arc_representation: torch.Tensor,
        child_arc_representation: torch.Tensor,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(child_arc_representation, head_arc_representation)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        decode_func = self._greedy_decode if self.training or not self.mst_decoding else self._mst_decode

        predicted_heads, predicted_deprels, predicted_confidences = decode_func(
            head_tag_representation, child_tag_representation, attended_arcs, mask
        )

        return predicted_heads, predicted_deprels, attended_arcs, predicted_confidences

    def _construct_loss(
        self,
        heads_logits: torch.Tensor,
        heads: torch.Tensor,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        deprels: torch.Tensor,
        mask: torch.BoolTensor,
        confidences: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Args:
            heads_logits : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
                a distribution over attachments of a given word to all other words.
            heads : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length).
                The indices of the heads for every word.
            deprels : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length).
                The dependency labels of the heads for every word.
            mask : `torch.BoolTensor`, required.
                A mask of shape (batch_size, sequence_length), denoting unpadded
                elements in the sequence.

        Returns:
            arc_nll : `torch.Tensor`, required.
                The negative log likelihood from the arc loss.
            tag_nll : `torch.Tensor`, required.
                The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = heads_logits.shape
        device = get_device_of(heads_logits)

        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, device).unsqueeze(1)
        # index matrix with shape (batch, sequence_length)
        # TODO: child_index: (batch, seq) -> (1, seq)
        timestep_index = get_range_vector(sequence_length, device)
        child_index = (
            timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        )

        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
            masked_log_softmax(heads_logits, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)

        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, heads]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, deprels]

        # weight the loss with confidences.
        arc_loss = arc_loss * confidences
        tag_loss = tag_loss * confidences

        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).
        Args:
            head_tag_representation : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            child_tag_representation : `torch.Tensor`, required
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            attended_arcs : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
                a distribution over attachments of a given word to all other words.
            # Returns
            predicted_heads : `torch.Tensor`
                A tensor of shape (batch_size, sequence_length) representing the
                greedily decoded heads of each word.
            predicted_deprels : `torch.Tensor`
                A tensor of shape (batch_size, sequence_length) representing the
                dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-numpy.inf)
        )

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, predicted_heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, predicted_heads
        )
        # shape (batch_size, sequence_length)
        _, predicted_deprels = torch.max(head_tag_logits, 2)

        predicted_confidences = torch.ones_like(predicted_heads)
        if not self.training:
            heads_p, _ = torch.max(attended_arcs.softmax(dim=-1), dim=-1)
            deprels_p, _ = torch.max(head_tag_logits.softmax(dim=-1), dim=-1)
            predicted_confidences = heads_p*deprels_p

        return predicted_heads, predicted_deprels, predicted_confidences

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Args:
            head_tag_representation : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            child_tag_representation : `torch.Tensor`, required
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            attended_arcs : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
                a distribution over attachments of a given word to all other words.

        Returns:
            predicted_heads : `torch.Tensor`
                A tensor of shape (batch_size, sequence_length) representing the
                greedily decoded heads of each word.
            predicted_deprels : `torch.Tensor`
                A tensor of shape (batch_size, sequence_length) representing the
                dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(
            0, 3, 1, 2
        )

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_heads = []
        predicted_deprels = []
        predicted_confidences = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            # NOTE: mistake in allennlp-model TODO: 长度，句子不登场，pad？？？
            # scores[0, :] = 0
            scores[:, 0] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            instance_confidences = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
                instance_confidences.append(scores[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            instance_confidences[0] = 0.
            predicted_heads.append(instance_heads)
            predicted_deprels.append(instance_head_tags)
            predicted_confidences.append(instance_confidences)
        return (
            torch.from_numpy(numpy.stack(predicted_heads)).to(batch_energy.device),
            torch.from_numpy(numpy.stack(predicted_deprels)).to(batch_energy.device),
            torch.from_numpy(numpy.stack(predicted_confidences)).to(batch_energy.device),
        )

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Args:
            head_tag_representation : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            child_tag_representation : `torch.Tensor`, required
                A tensor of shape (batch_size, sequence_length, tag_representation_dim),
                which will be used to generate predictions for the dependency tags
                for the given arcs.
            head_indices : `torch.Tensor`, required.
                A tensor of shape (batch_size, sequence_length). The indices of the heads
                for every word.

        Returns:
            head_tag_logits : `torch.Tensor`
                A tensor of shape (batch_size, sequence_length, num_head_tags),
                representing logits for predicting a distribution over tags
                for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size, 1)
        range_vector = torch.arange(0, batch_size, device=head_tag_representation.device).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )

        return head_tag_logits

    def _get_mask_for_eval(
        self, mask: torch.BoolTensor, confidences: torch.FloatTensor
    ) -> torch.BoolTensor:
        """
        Args:
            mask : `torch.BoolTensor`, required.
                The original mask.
            confidences : `torch.FloatTensor`, required.
                The confidences of label for the sequence.
        Returns:
            A new mask, where any indices equal to labels
            we should be ignoring are masked.
        """
        new_mask = mask
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset)

    def add_root_index(
        self,
        tokens: TextFieldTensors,
        postags: torch.LongTensor,
        heads: torch.Tensor,
        deprels: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets_tensor: torch.Tensor = tokens['tokens']['offsets']
        mask_tensor: torch.Tensor = tokens['tokens']['mask']
        token_characters_tensor: torch.Tensor = tokens['token_characters']['token_characters']
        batch_size, seq_len, char_len = token_characters_tensor.size()
        device = get_device_of(token_characters_tensor)

        root_index = torch.zeros((batch_size, 1, 2), dtype=torch.long, device=device)
        tokens['tokens']['offsets'] = torch.cat((root_index, offsets_tensor), dim=1)

        root_index = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        tokens['tokens']['mask'] = torch.cat((root_index, mask_tensor), dim=1)

        cls_chars = [
            self.vocab.get_token_index('C', 'token_characters'),
            self.vocab.get_token_index('L', 'token_characters'),
            self.vocab.get_token_index('S', 'token_characters'),
        ]
        root_index = torch.tensor([cls_chars for i in range(batch_size)], dtype=torch.long, device=device)
        root_index = torch.cat((
            root_index,
            torch.zeros((batch_size, char_len-3), dtype=torch.long, device=device)), dim=-1)
        tokens['token_characters']['token_characters'] = torch.cat(
            (root_index.unsqueeze(1), token_characters_tensor), dim=1)

        root_index = torch.full(
            (batch_size, 1), self.vocab.get_token_index('@@UNKNOWN@@', 'pos'),
            dtype=torch.long, device=device)
        postags = torch.cat((root_index, postags), dim=1)

        if heads is not None:
            root_index = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            heads = torch.cat((root_index, heads), dim=1)
        if deprels is not None:
            root_index = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            deprels = torch.cat((root_index, deprels), dim=1)
        if confidences is not None:
            root_index = torch.ones((batch_size, 1), device=device)
            confidences = torch.cat((root_index, confidences), dim=1)

        return postags, heads, deprels, confidences
