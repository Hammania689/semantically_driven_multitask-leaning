## Perl Wordnet Similarity Measure Installation

To DO
- [ ] Commit and push Docker Image.
- [ ] Add Wordnet to path
- [ ] Can the similarity module be added to path as well or atleast a script call?


Enusre you have Perl and Cpan installed.

You'll probably come accross errors while builing:
 
To install Digest (more info [here][digest]):
`cpan> install Digest::MD5
cpan> install Digest::SHA1`


Install the following packages (see post [here][package_post]):
`sudo apt-get install tcl-dev tk-dev mesa-common-dev libjpeg-dev libtogl-dev`

Any TCL related issues (e.g ` error: ‘Tcl_Interp’ has no member named ‘result’` - reference [here][tcl_post] )

In the stubs.c file (which will be found in the WordNet-3.0/srcdirectory include the following directive)

`#define USE_INTERP_RESULT 1`
before the 
`#include <tcl.h>`
and the do the make

## Folder that you wish to use can change in Docker I prefer /usr on my own machine I prefer /tmp until all files are extracted then I move over build libraries to /opt
    cd /usr
     wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz
     wget https://cpan.metacpan.org/authors/id/J/JR/JRENNIE/WordNet-QueryData-1.49.tar.gz
     wget https://cpan.metacpan.org/authors/id/T/TP/TPEDERSE/Text-Similarity-0.10.tar.gz
     wget https://cpan.metacpan.org/authors/id/T/TP/TPEDERSE/WordNet-Similarity-2.07.tar.gz
 
    Then unpack each one:
 
     tar -zxvf WordNet-3.0.tar.gz
     tar -zxvf WordNet-QueryData-1.49.tar.gz
     tar -zxvf Text-Similarity-0.10.tar.gz
     tar -zxvf WordNet-Similarity-2.07.tar.gz
 
    Install WordNet:
 
     cd /usr/WordNet-3.0
     ./configure
     make
     su
     make install
     exit
 
    Installing QueryData and Similarity:
 
     cd /usr/WordNet-QueryData-1.49
     perl Makefile.PL
     make
     make test
     su
     make install
     exit
 
     cd /usr/Text-Similarity-0.10
     perl Makefile.PL
     make
     make test
     su
     make install
     exit
 
     cd /usr/WordNet-Similarity-2.05
     perl Makefile.PL
     make
     make test
     su
     make install
     exit


To run script similarity measure use a similar format:  
`perl similarity.pl --type 'WordNet::Similarity::lesk'  chair#n#1 seat#n#1`


For more information refer to documentation or simply use this command:
`perl similarity.pl --h`

## To add to path to access systemwide. Add the following to your .bashrc
`export PATH="$PATH:/usr/WordNet-Similarity-2.07/utils"`

Adopted from author's original instructions [here][instructions]

[instructions]: [https://metacpan.org/source/TPEDERSE/WordNet-Similarity-2.07/INSTALL]
[digest]: [https://www.livejournal.com/doc/server/lj.install.perl_setup.modules.html]


[tcl_post]: [https://askubuntu.com/questions/372480/tcl-error-while-installing-wordnet/373364]
[package_post]: [https://www.linuxquestions.org/questions/linux-software-2/configuration-error-can%27t-find-tcl-configuration-definition-4175434069/]
