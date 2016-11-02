FIND=/usr/bin/find
ETAGS=/usr/bin/etags

tags ::
	rm -f TAGS \
	  && $(FIND) . -type f \
	     -name "*.java" \
	     -o -name "*.ts" \
	     -o -name "*.css" \
	  | xargs $(ETAGS) -a
