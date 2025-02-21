



## To run pyton file - be in conda env lausdLLM and then run below on bash.
## panel serve lausdLLM.py --port 5006 
                            (((( old )))) sudo -E /home/rchak007/anaconda3/envs/lausdLLM/bin/python -m panel serve lausdLLM.py --port 5006

## Then you have open in Browser for it to run - 
##     http://localhost:5006/lausdLLM


### create python file - from Jupyter notebook - 
## jupyter nbconvert --to script lausdLLM.ipynb
to create lausdLLM.py


### to open from jUpyter lab:


### conda activate lausdLLM
### jupyter lab

### use URL - http://127.0.0.1:8888/lab?token=c23c3b1cc47f9e0d178eabb1059241175607cb1b50f1cf6c
[I 2025-02-05 11:54:56.426 ServerApp] Jupyter Server 2.15.0 is running at:
[I 2025-02-05 11:54:56.426 ServerApp] http://localhost:8888/lab?token=c23c3b1cc47f9e0d178eabb1059241175607cb1b50f1cf6c
[I 2025-02-05 11:54:56.426 ServerApp]     http://127.0.0.1:8888/lab?token=c23c3b1cc47f9e0d178eabb1059241175607cb1b50f1cf6c
[I 2025-02-05 11:54:56.427 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-02-05 11:55:00.060 ServerApp]

    To access the server, open this file in a browser:
        file:///home/rchak007/.local/share/jupyter/runtime/jpserver-2312-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=c23c3b1cc47f9e0d178eabb1059241175607cb1b50f1cf6c
        http://127.0.0.1:8888/lab?token=c23c3b1cc47f9e0d178eabb1059241175607cb1b50f1cf6c
WARNING: You don't seem to have any mimeinfo.cache files.
Try running the update-desktop-database command. If you
don't have this command you should install the
desktop-file-utils package. This package is available from
http://freedesktop.org/wiki/Software/desktop-file-utils/




## ##############################################################################
## to do - 2/7/25
# off track vs on-track wrong answer - 
--- User:
can i report BEREAVEMENT on off-track days
ChatBot:
Yes, you can report BEREAVEMENT on off-track days.
# list all core is off - need to adjust chunks.. 

## half pay wage types - it gave only from workers comp section but did not from IL section
## - get all the chucks to show on retrieval so to debug more.




## ############## Common chain_type Values
"stuff" (Most Basic)

Loads all retrieved documents into the LLM prompt directly.
Fastest but may fail if documents are too long for the model’s context limit.
"map_reduce" (Summarizes First, Then Answers)

Summarizes individual chunks first, then combines results.
Good for large documents but slower.
"refine" (Step-by-Step Improvement)

Refines its response iteratively with each document.
Best when precise details matter, but very slow.

## qa is created using LangChain’s ConversationalRetrievalChain, so it returns a chatbot-like response that includes retrieved documents.


## #####################    important lessons

## chunking - RecursiveTextSplitter - does basic level more preserving doc formats.
##            TokenSplitter - did not use but normally use  RecursiveTextSplitter followed by TokenSplitter
##     but main challenges is because this word doc is more like RequirementS doc and not like true text document
##        so additionally we parse tables and bullets points first with Function - read_extract_Langdoc
##         then now we are trying dynamic chunking due to nature of this Requirements document. For us here we need each att/abs code section as 1 chunk.
   