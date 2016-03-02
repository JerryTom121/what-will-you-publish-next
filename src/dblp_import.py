# this script imports the dblp dataset and processes it

import cPickle as pickle
import argparse,gc
from lxml import etree
import HTMLParser
import re
from unidecode import unidecode
from utils2 import pubstring,authorstring,strip_author,progress_printer
import math,time

# parse command line arguments
parser = argparse.ArgumentParser(description='DBLP import script')
parser.add_argument('-f','--filename',required = True,
        help='Path and filename for dblp file. If file ending is .pkl, data will be read from pickled file directly (e.g. to re-create the output)')
parser.add_argument('-o','--output_path',required = True,
        help='Path for output file(s)')
parser.add_argument('-n','--number_of_lines',default=0,
        help='Number of lines to parse in file', type=int)
parser.add_argument('-c','--chunk_number',default=False,
        help="For distributed reading: number of this reader (don't use, artifact)", type=int)
parser.add_argument('-m','--number_of_chunks',default=False,
        help="For distributed reading: number of chunks (don't use, artifact)]", type=int)
parser.add_argument('-w','--without_authors_and_year',default=False,
        help='do not include authors in training data')
        
args = parser.parse_args()

filename=args.filename
number_of_lines=args.number_of_lines
output_path=args.output_path

# special case for distributed reading (don't use, artifact)
if args.chunk_number or args.number_of_chunks:
    assert isinstance( args.chunk_number, int )
    assert isinstance( args.number_of_chunks, int )
    assert args.chunk_number>0
    assert args.chunk_number<=args.number_of_chunks
    assert args.number_of_chunks>1
    assert number_of_lines==0
    
output_filename=output_path+"/dblp-plaintext.txt"
pickle_filename=output_path+"/dblp-data.pkl"

# special case for distributed reading (don't use, artifact)
if args.number_of_chunks:
    output_filename=False #can't produce output from part of the file
    pickle_filename=output_path+"/dblp-data_chunk_"+str(args.chunk_number)+"of"+str(args.number_of_chunks)+".pkl"

# define some functions we need later on

# reset stuff
def reset():
    global publications,author_pubs,pubid,journals_or_publishers
    publications=[]
    journals_or_publishers=[]
    author_pubs={}
    pubid=0

# process a single XML element
def process_element(child):
    global pubid,publications,author_pubs,journals_or_publishers
    # we don't read www tags
    if child.tag!="www":
        fine=False
        if child.tag=="book":
            entry_type="b"
        elif child.tag=="incollection":
            if 'publtype' in child.attrib:
                if child.attrib['publtype']=="encyclopedia entry":
                    return
            entry_type="i"
        elif child.tag=="article" or child.tag=="inproceedings":
            entry_type="a"
        else:
            return
        #print child.tag
        authors=[]
        year=None
        journal_or_publisher=None
        title=None
        # disable garbage collection (can help with appending to long lists)
        gc.disable()
        # read all sub elements of this element
        for c2 in child:
            #print "\t",c2.tag,c2.text
            if c2.tag=="author":
                authors.append(c2.text)
            elif c2.tag=="title":
                title=c2.text
            elif c2.tag=="year":
                year=c2.text
            elif c2.tag=="journal":
                journal_or_publisher=c2.text
            elif c2.tag=="booktitle":
                journal_or_publisher=c2.text
            elif c2.tag=="publisher":
                journal_or_publisher=c2.text
        if len(authors)>0 and title!=None and journal_or_publisher!=None and year!=None:
            # add journal
            try:
                journal_or_publisher_id=journals_or_publishers.index(journal_or_publisher)
            except ValueError:
                journals_or_publishers.append(journal_or_publisher)
                journal_or_publisher_id=len(journals_or_publishers)-1
            # add publication
            # entry_type,authors,title,journal_or_publisher_id,year
            publications.append([entry_type,authors,title,journal_or_publisher_id,year])
            fine=True
        
        # add authors
        for author in authors:
            # using try/except is about a gazillion times faster then checking if author is in .keys()
            try:
                author_pubs[author].append(pubid)
            except KeyError:
                author_pubs[author]=[]
                author_pubs[author].append(pubid)
        gc.enable()
        if fine:
            pubid+=1
            return True

# reads in the data from an XML file
def read_data():
    global pubid,publications,author_pubs,journals_or_publishers,filename,number_of_lines
    h = HTMLParser.HTMLParser()

    parsed_elements=0

    reset()
    inputbuffer = ''
    tokens=["article","incollection","book ", "inproceedings"]
    start=time.time()
    with open(filename,'rb') as inputfile:
        append = False
        start_token=-1
        # we go through the file line by line, as many python XML parsers choke on the sheer size of the DBLP file (1.9 GB), and build up a huge memory footprint
        for line in inputfile:
            if parsed_elements<fromline:
                parsed_elements+=1
                continue
            # this assumes that the xml is valid, i.e., nested properly!
            if any(["<"+x in line for x in tokens]):
                inputbuffer = line
                append = True
            elif any(["</"+x in line for x in tokens]) and append:
                inputbuffer += line
                append = False
                buf=h.unescape(inputbuffer.replace("&lt;"," smaller ").replace("&gt;"," greater ")).replace("&"," and ")
                try:
                    process_element(etree.fromstring(buf))
                except:
                    print ""
                    print buf
                    raise
                    
                inputbuffer = None
            elif append:
                inputbuffer += line
                
            progress_printer(parsed_elements-fromline,number_of_lines)
            
            parsed_elements+=1
            
            # this realizes a "soft" cutoff, lines will be read until current element is processed and buffer is empty
            if parsed_elements>=number_of_lines+fromline and inputbuffer==None:
                #time.sleep(0.1) # let progress printer catch up, so that people won't be confused by non-100% reading
                #progress_printer(parsed_elements-fromline,number_of_lines)
                break
print "\r"
    
######################################
## now the actual script

# standard case: we read in an xml file
if filename[-4:]!=".pkl":
    # read in data
    print "counting number of lines"
    with open(filename) as f:
        for num_lines_in_input_file, l in enumerate(f):
            pass
    num_lines_in_input_file+=1
    print "There are "+str("{:,}".format(num_lines_in_input_file))+" lines in input file"

    fromline=0
    if number_of_lines==0:
        number_of_lines=num_lines_in_input_file    
    
    # distributed reading, dont use
    if args.chunk_number:
        number_of_lines=math.ceil(num_lines_in_input_file/args.number_of_chunks)
        fromline=math.floor((num_lines_in_input_file/args.number_of_chunks)*(args.chunk_number-1))
        print "This script processes chunk %d of %d, i.e., it will start at line %d and process %d lines"%(args.chunk_number,args.number_of_chunks,fromline,number_of_lines)
    else:
        print "This script will procress %d of %d lines in input file, starting at the beginning"%(number_of_lines,num_lines_in_input_file)
    
    reset()
    print "reading data"
    read_data()

    print "\rdone"
    
    print "found "+str(len(publications))+ " publications ("+str(len(publications)/float(number_of_lines))+" per parsed input line)"
    print "found "+str(len(author_pubs))+ " authors"
    print "found "+str(len(journals_or_publishers))+ " journal names/publishers"


    print "dumping variables" # to pickle file
    with open(pickle_filename,"w") as f:
        pickle.dump([publications,journals_or_publishers,author_pubs],f)
# we can also read a previously written pickle file
else:
    print "loading pickled results"
    publications,journals_or_publishers,author_pubs=pickle.load(open(filename))


# write the output in the desired format 
if output_filename:
    print "writing output"
    write_after_n_authors=1000
    buf=""
    n=0
    f=open(output_filename,"w")
    for author in author_pubs:
        auth=strip_author(author)
        if not args.without_authors_and_year:
            buf+="--beginauthor-- "+auth+"\n"
        buf+=authorstring(author,author_pubs,publications,journals_or_publishers,without_authors_and_year=args.without_authors_and_year).lower()
        if not args.without_authors_and_year:
            buf+="--endauthor--\n"
        if n==write_after_n_authors:
            f.write(unidecode(buf).encode('utf8'))
            buf=""
            n=0
        else:
            n+=1 
    f.write(unidecode(buf).encode('utf8'))
    f.close()
else:
    print "when all output pickle files are ready, combine them using combine_import_chunks.py"

print "done"





