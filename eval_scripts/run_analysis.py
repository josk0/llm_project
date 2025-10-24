import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# In[1]:


import json


# In[2]:


with open("results/out.json", 'r') as f:
    res = json.load(f)


# In[3]:


orig_model_res = res['orig']
fine_model_res = res['finetuned']


# In[4]:




# In[5]:





plt.hist(orig_model_res['jaccard'], alpha = 0.5, color = 'blue', label = 'Jaccard simularity, original model, mean is {:.4f}, std {:.4f}'.format(np.mean(orig_model_res['jaccard']), np.std(orig_model_res['jaccard'])))
plt.axvline(np.mean(orig_model_res['jaccard']), color = 'blue', alpha = 0.5)

plt.hist(fine_model_res['jaccard'], alpha = 0.5, color = 'green', label = 'Jaccard simularity, finetuned model, mean is {:.4f}, std {:.4f}'.format(np.mean(fine_model_res['jaccard']), np.std(fine_model_res['jaccard'])))
plt.axvline(np.mean(fine_model_res['jaccard']), color = 'green', alpha = 0.5)
plt.legend()
plt.savefig("jaccard.png", bbox_inches = "tight")
plt.show()


# In[6]:


plt.hist(orig_model_res['compression'], alpha = 0.5, color = 'blue', label = 'Compression simularity, original model, mean is {:.4f}'.format(np.mean(orig_model_res['compression'])))
plt.axvline(np.mean(orig_model_res['compression']), color = 'blue', alpha = 0.5)

plt.hist(fine_model_res['compression'], alpha = 0.5, color = 'green', label = 'Compression simularity, finetuned model, mean is {:.4f}'.format(np.mean(fine_model_res['compression'])))
plt.axvline(np.mean(fine_model_res['compression']), color = 'green', alpha = 0.5)
plt.legend()
plt.savefig("compression.png", bbox_inches = "tight")
plt.show()


# In[8]:


plt.hist(orig_model_res['spacy_sim'], alpha = 0.5, color = 'blue', label = 'Spacy simularity, original model, mean is {:.4f}'.format(np.mean(orig_model_res['spacy_sim'])))
plt.axvline(np.mean(orig_model_res['spacy_sim']), color = 'blue', alpha = 0.5)

plt.hist(fine_model_res['spacy_sim'], alpha = 0.5, color = 'green', label = 'Spacy simularity, finetuned model, mean is {:.4f}'.format(np.mean(fine_model_res['spacy_sim'])))
plt.axvline(np.mean(fine_model_res['spacy_sim']), color = 'green', alpha = 0.5)
plt.legend()
plt.savefig("spacy_sim.png", bbox_inches = "tight")
plt.show()


# In[9]:


orig_model_our = pd.DataFrame.from_dict(orig_model_res['burrows']).transpose()['Our corpus']
fine_model_our = pd.DataFrame.from_dict(fine_model_res['burrows']).transpose()['Our corpus']

plt.hist(orig_model_our, alpha = 0.5, color = 'blue', label = 'Burrows delta, original model, mean is {:.4f}, std {:.4f}'.format(np.mean(orig_model_our), np.std(orig_model_our)))
plt.axvline(np.mean(orig_model_our), color = 'blue', alpha = 0.5)

plt.hist(fine_model_our, alpha = 0.5, color = 'green', label = 'Burrows delta, finetuned model, mean is {:.4f}, std {:.4f}'.format(np.mean(fine_model_our), np.std(fine_model_our)))
plt.axvline(np.mean(fine_model_our), color = 'green', alpha = 0.5)
plt.legend()

plt.title("Burrows delta for our corpus (papers) vs orig model/finetuned model")
plt.savefig("burrows_delta_our_corpus.png", bbox_inches = "tight")
plt.show()


# In[10]:


orig_model_our = pd.DataFrame.from_dict(orig_model_res['burrows']).transpose()['LLM-corpus']
fine_model_our = pd.DataFrame.from_dict(fine_model_res['burrows']).transpose()['LLM-corpus']

plt.hist(orig_model_our, alpha = 0.5, color = 'blue', label = 'Burrows delta, original model, mean is {:.4f}, std {:.4f}'.format(np.mean(orig_model_our), np.std(orig_model_our)))
plt.axvline(np.mean(orig_model_our), color = 'blue', alpha = 0.5)

plt.hist(fine_model_our, alpha = 0.5, color = 'green', label = 'Burrows delta, finetuned model, mean is {:.4f}, std {:.4f}'.format(np.mean(fine_model_our), np.std(fine_model_our)))
plt.axvline(np.mean(fine_model_our), color = 'green', alpha = 0.5)
plt.legend()
plt.title("Burrows delta for broad LLM-generated corpus vs orig model/finetuned model")
plt.savefig("burrows_delta_llm_corpus.png", bbox_inches = "tight")
plt.show()


