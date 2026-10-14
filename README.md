Capstone Project Checkpoints

Scenario:
Research papers 

Objective:

Each capstone has different objective ranging from 
- identifying whether retrieval is required 
- whether to use keyword based retrieval, semantic retrieval or a mix of both forms of retrieval
- Evaluating the retrieved content and generated answer against a benchmark
- Decomposing single retrieval into multiple retrieval steps
- Decide if more retrieval is required to answer
- Demonstrating and preventing security attacks
- to measuring cost of models for different activities and choosing a optimised model for each activity in retrieval and answering


How to run:

To run capstone checkpoint, follow the below instructions:

OpenAPI Key setup:
    Type touch .env
    Open .env file and add the following:
    OPENROUTER_API_KEY= "key provided in email by support team"

Python environment setup:
    python3 -m venv venv

Activating the python environment:
    source venv/bin/activate

Core packages installation:
    Type  python3 install dotenv
    Type  python3 install langchain_core
    Type  python3 install langchain_openai
    Type  python3 install langchain_chroma
    Type  python3 install langgraph
    Type  python3 install langchain_text_splitters
    Type  python3 install rank_bm25
    Type  python3 install pypdf


Run Checkpoint:
   Create a folder named chroma_db within respective capstone folder
   
   For capstone 1 through 5, cd into the respective capstone checkpoint folder and then type 
   python3 Solution.py

   For capstone 6, cd into the respective folder and then type
   python3 CostMeasurement_Solution.py 
   python3 Security_Solution.py

   For capstone 7, cd into the respective folder and then type
   python3 EfficientAgent_Solution.py 
   python3 SafeAgent_Solution.py





