# crosslingual-DepPar

crosslingual dependency parsing

## preprocessing

### opus data
> opus_read -d Europarl -s en -t de -w en-de.align -ln -wm moses -cm ' ||| '  

### fast align  
> ./fast_align -i en-de.align -d -o -v -I 10 > forward.align  
  
> ./fast_align -i en-de.align -d -o -v -r -I 10 > reverse.align  
  
> ./atools -i forward.align -j reverse.align -c grow-diag-final-and > en-de.token.align  

### udpipe2 tagging
> curl -F data=@en.txt -F model=english-ewt-ud-2.6-200830 -F input=horizontal -F output=conllu -F tagger= http://lindat.mff.cuni.cz/services/udpipe/api/process | PYTHONIOENCODING=utf-8 python -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])" > en.tag.txt  



## requirements
`allennlp`  
`opustools`  
`jsonlines`
