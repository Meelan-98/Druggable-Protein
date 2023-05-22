# Druggable Proteins Binary Classification

### Create the environment

```
    python3 -m venv env 
```

### Activate the environment

```
    mac : source env/bin/activate
    
    windows : .\env_name\Scripts\activate
```

### Install the requirements

```
    pip3 install -r requirements.txt
```

### Run the classifier

```
    python3 classifier-tool.py TR_neg_SPIDER TR_pos_SPIDER TS_neg_SPIDER TS_pos_SPIDER
```

### Run the report generator

```
    python data-profile.py
```