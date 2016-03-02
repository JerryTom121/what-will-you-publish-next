#!/usr/bin/python
import math,sys,cPickle as pickle,time,re,random

from unidecode import unidecode

def strip_author(author):
    auth=author.split(" ")[-1]
    if re.match("\d\d\d\d", auth):
        auth=author.split(" ")[-2]
    return auth.lower()

# pads special characters by spaces (otherwise, "blah" and "blah?" and "blah." are three distinct words occupying space in our dictionary
def pad_special_chars(string_in):
    chars=[":",",","-","?","(",")","/",".","*","$","%","!","&","#","=","+","^"]
    for char in chars:
        string_in=string_in.replace(char," "+char+" ")
    return string_in
# remove/repalce some special characters
def replace_special_chars(string_in):
    repl=[("<","("),(">",")"),("{","("),("}",")"),("[","("),("]",")"),("'",""),('"',""),("\\",""),(";",",")]
    for rep in repl:
        string_in=string_in.replace(rep[0],rep[1])
    return string_in
    
# we don't want to learn numbers
def remove_numbers(string_in):
    return re.sub('[\d]+[\.]*[\d]*',"N",string_in)

# some journal/conference title abbreviations are ugly, and unnecessary (abbreviation just occupies another dictionary entry)
def journal_replace(string_in):
    repl=[
        ("J.","Journal"),
        ("I.","International"),
        ("Int.","International"),
        ("Approx.","Approximative"),
        ("Sel.","Selected"),
        ("Sel.","Selected"),
        ("Math.","Mathematics"),
        ("Meth.","Methods"),
        ("Sel.","Selected"),
        ("Adv.","Advanced"),
        ("Biomed.","Biomedical"),
        ("Proc.","Proceedings"),
        ("Inf.","Information"),
        ("Inform.","Information"),
        ("Sci.","Science"),
        ("Hum.","Human"),
        ("Eng.","Engineering"),
        ("Interact.","Interaction"),
        ("Trans.","Transactions"),
        ("Autom.","Automation"),
        ("Emb.","Embedded"),
        ("Sys.","Systems"),
        ("Theor.","Theoretical"),
        ("Med.","Medical"),
        ("Graph.","Graphics"),
        ("Imag.","Imaging"),
        ("Artif.","Artificial"),
        ("Intell.","Intelligent"),
        ("Syst.","Systems"),
        ("Bull.","Bulletin"),
        ("Mag.","Magazine"),
        ("Symb.","Symbolic"),
        ("Log.","Logic"),
        ("Model.","Modelling"),
        ("Interact.","Interacive"),
        ("Edu.","Education"),
        ("T.","Transactions"),
        ("Front.","Frontiers"),
        ("Opt.","Optimization"),
        ("Softw.","Software"),
        ("Vis.","Vision"),
        ("Ann.","Annals"),
        ("Knowl.","Knowledge"),
        ("Technol.","Technology"),
        ("Neurosci.","Neuroscience"),
        ("Neurosc.","Neuroscience"),
        ("Netw.","Network"),
        ("Electron.","Electronic"),
        ("Res.","Research"),
        ("Retr.","Retrieval"),
        ("Rel.","Reliability"),
        ("Mod.","Models"),
        ("Optim.","Optimization"),
        ("Sci.","Scientific"),
        ("Commun.","Communications"),
        ("Surv.","Survey"),
        ("Bio.","Biology"),
        ("Simul.","Simuation"),
        ("Program.","Programming"),
        ("Prog.","Programming"),
        ("Lang.","Languages"),
        ("Algebr.","Algebraic"),
        ("Soc.","Social"),
        ("Cyberpsy.","Cyberpsychology"),
        ("Mach.","Machine"),
        ("Anal.","Analysis"),
        ("Des.","Designs"),
        ("Dynam.","Dynamics"),
        ("Cybern.","Cybernetics"),
        ("Intellig.","Intelligent"),
        ("Transport.","Transportation"),
        ("Arch.","Archive"),
        ("Manag.","Managment"),
        ("Strat.","Strategic"),
        ("Let.","Letters"),
        ("Lett.","Letters"),
        ("Visualiz.","Visualization"),
        ("Methodol.","Methodologies"),
        ("Techn.","Technical"),
        ("Zeitschr.","Zeitschrift"),
        ("Process.","Processing"),
        ("Wirtschaftsinform.","Wirtschaftsinformatik"),
    ]
    for rep in repl:
        string_in=string_in.replace(rep[0],rep[1])
    return string_in

# creates a string from all publications of an author
def pubstring(pub,journals_or_publishers,without_authors_and_year=False):
    mystr=""
    #entry_type,authors,title,journal_or_publisher_id,year
    first=True
    # add authors
    if not without_authors_and_year:
        for a in pub[1]:
            if not first:
                mystr=mystr+" , "
            # only use authors last names. dictionary is already huge!
            auth=strip_author(a)
            mystr=mystr+auth
            first=False
        mystr+=' ; '
    # title, but without trailing "."
    if pub[2][-1]==".":
        mystr+=pad_special_chars(replace_special_chars(remove_numbers(pub[2][:-1])))+' ; '
    else:
        mystr+=pad_special_chars(replace_special_chars(remove_numbers(pub[2])))+' ; '
    
    # add journal/conference/publisher
    if pub[0]=='a' or pub[0]=='i':
        mystr=mystr+" In: "
    mystr=mystr+" "+pad_special_chars(replace_special_chars(remove_numbers(journal_replace(journals_or_publishers[pub[3]]))))
    # add year
    if not without_authors_and_year:
        mystr+=" ; "+pub[4]
    return mystr
    
# creates a string for a given author
def authorstring(name,author_pubs,publications,journals_or_publishers,only_n_pubs=None,remove_years=None,without_authors_and_year=False):
    mystr=""
    pubs= sorted(author_pubs[name], key=lambda x: publications[x][4])
    if not remove_years==None:
        pubs_reduced=[]
        for pub in pubs:
            if int(publications[pub][4]) not in remove_years:
                pubs_reduced.append(pub)
        if len(pubs_reduced)>3:
            pubs=pubs_reduced
    if not only_n_pubs==None and len(pubs)>only_n_pubs:
        pubs=[pubs[i] for i in random.sample(xrange(len(pubs)),only_n_pubs)]
    for pub in pubs:
        mystr+="--beginentry-- "
        if not without_authors_and_year:
            mystr+=strip_author(name)+" "
        mystr+=pubstring(publications[pub],journals_or_publishers,without_authors_and_year)
        if not without_authors_and_year:
            mystr+=" --endentry--"
        mystr+="\n"
        
    return mystr

# nice progress printer
def progress_printer(current_number,total_number):
    """
    This function does nothing but displaying a fancy progress bar :)
    """
    global anim_state,pp_last_print_time

    if not 'anim_state' in globals():
        anim_state=0
    
    if 'pp_last_print_time' not in globals():
        printme=True
    elif time.time()-pp_last_print_time>0.1:
        printme=True
    else:
        printme=False

    if printme:
        anim=["[*     ]","[ *    ]","[  *   ]","[   *  ]","[    * ]","[     *]","[    * ]","[   *  ]","[  *   ]","[ *    ]"]
        #anim=["|  "," | ","  |"," | "]
        if total_number!=None and total_number!=0:
            progress=str(int((float(current_number)/total_number)*100))+"%"
        else:
            progress=" *working hard* ("+str("{:,}".format(current_number))+" elements processed)"
        print "\r"+anim[anim_state]+" "+progress,
        anim_state=(anim_state+1)%len(anim)
        pp_last_print_time=time.time()
        sys.stdout.flush()
