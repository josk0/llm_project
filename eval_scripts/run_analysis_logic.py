import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tqdm
from itertools import chain
import spacy
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import numpy as np

# In[2]:


with open("out/earthy-sponge-20-final_logic.json", 'r') as f:
    res = json.load(f)


# In[3]:


res.keys()


# In[4]:




df = pd.DataFrame()

df['ground_truth'] = res["ground_truth"]
df['responses_ft'] = res["responses_ft"]
df['responses_orig'] = res["responses_orig"]
df['depth'] = res["inputs_depth"]


# In[5]:


df['ground_truth_true'] = df['ground_truth'].apply(lambda x: x=="true")
df['ground_truth_false'] = df['ground_truth'].apply(lambda x: x=="false")
df['ground_truth_uncertain'] = df['ground_truth'].apply(lambda x: x=="uncertain")


# In[6]:




le = LabelEncoder()
df['ground_truth_encoded'] = le.fit_transform(df["ground_truth"])
# df['ground_truth_encoded']


# In[7]:


test_len = len("You must then determine whether the statement is true or false.")


# In[8]:


df["index_reply_ft"] = df['responses_ft'].apply(lambda x: x.find("You must then determine whether the statement is true or false."))
df["index_reply_orig"] = df['responses_orig'].apply(lambda x: x.find("You must then determine whether the statement is true or false."))


# In[9]:


df['cut_resp_ft'] = df['responses_ft'].apply(lambda x: x[x.find("You must then determine whether the statement is true or false.") + test_len:])
df['cut_resp_orig'] = df['responses_orig'].apply(lambda x: x[x.find("You must then determine whether the statement is true or false.")+test_len:])


# In[10]:


nlp = spacy.load("en_core_web_sm") # Load a small English model

# df['testing'] = df['cut_resp_orig'].apply(lambda x:sent_tokenize(x))
df['testing'] = df['cut_resp_orig'].apply(lambda x:[y.text for y in nlp(x).sents])


# In[11]:




c = Counter(chain(*df['testing'].tolist()))


# In[12]:


c.most_common(100)


# In[13]:


df['conv_ft_true'] = df['cut_resp_ft'].apply(lambda x: "the statement is true" in x.lower() or "statement above is true" in x.lower())
df['conv_ft_false'] = df['cut_resp_ft'].apply(lambda x: "the statement is false" in x.lower() or "the statement is untrue" in x.lower() or "statement above is false" in x.lower())
df['conv_ft_uncertain'] = df['cut_resp_ft'].apply(lambda x: "the statement is uncertain" in x.lower() or "the statement is neither true nor false" in x.lower() or "statement above is uncertain" in x.lower() or "the statement is uncertain because it is neither true nor false" in x.lower())


df['conv_orig_true'] = df['cut_resp_orig'].apply(lambda x: "the statement is true" in x.lower() or "statement above is true" in x.lower())
df['conv_orig_false'] = df['cut_resp_orig'].apply(lambda x: "the statement is false" in x.lower() or "the statement is untrue" in x.lower() or "statement above is false" in x.lower())
df['conv_orig_uncertain'] = df['cut_resp_orig'].apply(lambda x: "the statement is uncertain" in x.lower() or "the statement is neither true nor false" in x.lower() or "statement above is uncertain" in x.lower() or "the statement is uncertain because it is neither true nor false" in x.lower())



df["unique_answer_ft"] = (df[["conv_ft_true", "conv_ft_false", "conv_ft_uncertain"]].sum(axis=1) == 1).values
df["unique_answer_orig"] = (df[["conv_orig_true", "conv_orig_false", "conv_orig_uncertain"]].sum(axis=1) == 1).values


df['ft_pred_name'] = df[["conv_ft_true", "conv_ft_false", "conv_ft_uncertain"]].idxmax(axis=1)
df['orig_pred_name'] = df[["conv_orig_true", "conv_orig_false", "conv_orig_uncertain"]].idxmax(axis=1)


mapper = {"conv_ft_true":"true", "conv_ft_false":"false", "conv_ft_uncertain":"uncertain", "conv_orig_true":"true", "conv_orig_false":"false", "conv_orig_uncertain":"uncertain"}
df['ft_pred'] = df['ft_pred_name'].apply(lambda x: mapper[x])
df['orig_pred'] = df['orig_pred_name'].apply(lambda x: mapper[x])



f1_finetuning = []
number_parsed_ft = []
for d in range(1, 7):
    df_subset = df[df['depth']==d]
    predictions_ft = le.transform(df_subset[df_subset["unique_answer_ft"]]['ft_pred'])
    number_parsed_ft.append(len(predictions_ft))
    ground_truth_ft = df_subset[df_subset["unique_answer_ft"]]['ground_truth_encoded']
    f1_finetuning.append(f1_score(predictions_ft, ground_truth_ft, average = 'macro'))
    print(d, classification_report(predictions_ft, ground_truth_ft))


plt.plot(range(1, 7), f1_finetuning)


plt.plot(range(1, 7), number_parsed_ft)


predictions_ft = le.transform(df[df["unique_answer_ft"]]['ft_pred'])
ground_truth_ft = df[df["unique_answer_ft"]]['ground_truth_encoded']

print(classification_report(predictions_ft, ground_truth_ft))


f1_orig = []
number_parsed_orig = []
for d in range(1, 7):
    df_subset = df[df['depth']==d]
    predictions_orig = le.transform(df_subset[df_subset["unique_answer_orig"]]['orig_pred'])
    number_parsed_orig.append(len(predictions_orig))
    ground_truth_orig = df_subset[df_subset["unique_answer_orig"]]['ground_truth_encoded']
    f1_orig.append(f1_score(predictions_orig, ground_truth_orig, average = 'macro'))
    print(d, classification_report(predictions_orig, ground_truth_orig))


# In[23]:




fig = plt.figure(figsize = (8,6))

plt.plot(range(1, 7), f1_orig, label = "F1 score for original model", color = 'blue')
plt.plot(range(1, 7), f1_finetuning, label = "F1 score for finetuned model", color = 'orange')


random_baseline_value = np.mean([f1_score(np.random.randint(0, 3, len(ground_truth_orig)), ground_truth_orig, average = "macro") for x in range(100)])

plt.axhline(random_baseline_value, label = 'random baseline', color = 'red')

predictions_orig = le.transform(df[df["unique_answer_orig"]]['orig_pred'])
ground_truth_orig = df[df["unique_answer_orig"]]['ground_truth_encoded']
plt.axhline(f1_score(predictions_orig, ground_truth_orig, average = 'macro'), label = 'avg f1, orig model', color = 'blue')

predictions_ft = le.transform(df[df["unique_answer_ft"]]['ft_pred'])
ground_truth_ft = df[df["unique_answer_ft"]]['ground_truth_encoded']
plt.axhline(f1_score(predictions_ft, ground_truth_ft, average = 'macro'), label = 'avg f1, finetuned model', color = 'orange')

plt.xlabel("Depth of the argument")
plt.ylabel("F1 score, macro")

plt.title("Complexity of the argument vs accuracy")
plt.legend()
plt.savefig("complexity_vs_f1.png", bbox_inches = 'tight')
plt.show()


# In[24]:


fig = plt.figure(figsize = (8, 6))
plt.plot(range(1, 7), number_parsed_orig, label = "Successfully parsed for original model")
plt.plot(range(1, 7), number_parsed_ft, label = "Successfully parsed for finetuned model")

plt.xlabel("Depth of the argument")
plt.ylabel("Number of succesfully parsed responses")

plt.title("Complexity of the argument vs number of parsed")
plt.legend()
plt.savefig("number_parsed.png", bbox_inches = 'tight')
plt.legend()


# In[112]:


predictions_orig = le.transform(df[df["unique_answer_orig"]]['orig_pred'])
ground_truth_orig = df[df["unique_answer_orig"]]['ground_truth_encoded']


# In[113]:


print(classification_report(predictions_orig, ground_truth_orig))
