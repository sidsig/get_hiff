import time
import os

while True:
	print "clearing lock dir"
	os.system("rm -r /homes/alexc/.theano/compiledir_Linux-2.6.32-431.3.1.el6.centos.plus.x86_64-x86_64-with-centos-6.5-Final-x86_64-2.6.6/lock_dir")
	print "sleeping..."
	time.sleep(5)

