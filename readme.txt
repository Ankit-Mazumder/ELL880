Architecture of the application:

User Interface:
	-> This consists of all the conferences with A* ranking.(Used list box for creating it).
	-> There will be a search bar which is used to search the research papers and their citation count.
	-> There will be two graph buttons in which graph gives the conference citation index for all the 100 percent papers and graph2 gives the conference citation index for 80 percent papers.
Scrappers:
	-> conference_scrapper.py which is used to get the list of conferences with A* ranking.
	-> events_scrapper.py which is used to get the list of events in each conference for the particular year.
	-> rp.py is used to get the research papers of the each event and their semantic scholar link.
	-> citation.py is used to get the number of citations for each paper.
Database:
	-> Finally, we will create the database with the schema(sno,paper,conference,code,year,citations).
-> Connecting the user interface to the backend for retrieving the research papers along with the citations.
-> Plotting the graph1 and graph2 using the data present in the database.
Update Button:
	-> When the user clicks the update button, it will run all the four scrappers and gets the new data and then it plots the graphs.


API:Semantic Scholar Academic Graph API
If we get the api, the implementation will be fast and efficient.

Note: We will take the research papers that are published in the last four years only.
conference portal link: http://portal.core.edu.au/conf-ranks/

list of popular journals along with the impact factors:

1.National Conference of the American Association for Artificial Intelligence	AAAI
2.International Joint Conference on Autonomous Agents and Multiagent Systems (previously the International Conference on Multiagent Systems, ICMAS, changed in 2000)	AAMAS
3.Association of Computational Linguistics	ACL
4.ACM Multimedia ACMMM
5.Automated Software Engineering Conference ASE
6.Architectural Support for Programming Languages and Operating Systems ASPLOS
7.Computer Aided Verification CAV
8.ACM Conference on Computer and Communications Security CCS
9.International Conference on Human Factors in Computing Systems CHI
10.Conference on Learning Theory COLT
11.Advances in Cryptology CRYPTO
12.IEEE Conference on Computer Vision and Pattern Recognition CVPR
13.ACM Conference on Economics and Computation EC
14.European Conference on Computer Vision ECCV
15.European Software Engineering Conference and the ACM SIGSOFT Symposium on the Foundations of Software Engineering (duplicate was listed as ESEC, removed from DB)	ESEC/FSE		
16.International Conference on the Theory and Application of Cryptographic Techniques	EuroCrypt		
17.IEEE Symposium on Foundations of Computer Science	FOCS	
18.International Symposium on High Performance Computer Architecture	HPCA	
19.International Conference on Automated Planning and Scheduling	ICAPS	
20.IEEE International Conference on Computer Vision	ICCV	
21.International Conference on Data Engineering	ICDE	
22.IEEE International Conference on Data Mining	ICDM	
23.International Conference on Learning Representations	ICLR
24.International Conference on Machine Learning	ICML	
25.International Conference on Software Engineering	ICSE	
26.International Joint Conference on Artificial Intelligence	IJCAI	
27.IEEE International Conference on Computer Communications	INFOCOM	
28.Information Processing in Sensor Networks	IPSN	
29.ACM International Symposium on Computer Architecture	ISCA	
30.IEEE/ACM International Symposium on Mixed and Augmented Reality	ISMAR	
31.ACM International Conference on Knowledge Discovery and Data Mining	KDD	
32.International Conference on the Principles of Knowledge Representation and Reasoning	KR	
33.IEEE Symposium on Logic in Computer Science	LICS	
34.ACM International Conference on Mobile Computing and Networking	MOBICOM	\
35.Usenix Network and Distributed System Security Symposium	NDSS	
36.Advances in Neural Information Processing Systems (was NIPS)	NeurIPS	
37.Usenix Symposium on Operating Systems Design and Implementation	OSDI
38.IEEE International Conference on Pervasive Computing and Communications	PERCOM	
39.ACM-SIGPLAN Conference on Programming Language Design and Implementation	
40.ACM Symposium on Principles of Distributed Computing	PODC	
41.ACM SIGMOD-SIGACT-SIGART Conference on Principles of Database Systems	PODS	
42.ACM-SIGACT Symposium on Principles of Programming Languages	POPL	
43.Real Time Systems Symposium	RTSS	
44.ACM Conference on Embedded Networked Sensor Systems	SENSYS	
45.ACM Conference on Applications, Technologies, Architectures, and Protocols for Computer Communication	SIGCOMM	
46.ACM SIG International Conference on Computer Graphics and Interactive Techniques	SIGGRAPH	
47.ACM International Conference on Research and Development in Information Retrieval	SIGIR	
48.Measurement and Modeling of Computer Systems	SIGMETRICS	
49.ACM Special Interest Group on Management of Data Conference	SIGMOD	
50.ACM/SIAM Symposium on Discrete Algorithms	SODA	
51.ACM SIGOPS Symposium on Operating Systems Principles	SOSP	
52.IEEE Symposium on Security and Privacy	SP	
53.ACM Symposium on Theory of Computing	STOC	
54.ACM Symposium on User Interface Software and Technology	UIST
55.Usenix Security Symposium	USENIX-Security	
56.International Conference on Very Large Databases	VLDB	
57.IEEE Conference on Virtual Reality and 3D User Interfaces	VR	
58.ACM International Conference on Web Search and Data Mining	WSDM	
59.International World Wide Web Conference	WWW

jan: architecture and user interface
feb: API's or scrappers
march: database and plotting the graphs.
April: debugging.

Notes of meeting:
-> Every events has workshops and other things asso
