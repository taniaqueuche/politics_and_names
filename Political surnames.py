#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import pandas as pd
from os import path
from PIL import Image

#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 16)


# In[38]:


import sys
print(sys.executable)


# In[48]:


#!<C:\Users\Tania\Anaconda3\python.exe>/python -m pip install wordcloud
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


#inspired by: https://www.datacamp.com/community/tutorials/wordcloud-python
#dataset: https://www.kaggle.com/aavigan/house-of-representatives-congress-116?select=house_members_116.csv


# # Formatting

# In[109]:


df = pd.read_csv('house_members_116.csv')
df.head()


# In[110]:


df['surname']=""
df = df[["name_id", "surname", "name", "state", "chamber", "current_party", "committee_assignments", "url"]]
df['surname']= df["name"].str.split("-", n = 1, expand = True) 
df.head()


# In[111]:


df_republicans = df.loc[df["current_party"]=="Republican"]
df_democrats = df.loc[df["current_party"]=="Democratic"]


# # Creating Wordclouds

# In[112]:


#creating a text block with all republican names
text_rep = " ".join(df_republicans for df_republicans in df_republicans.surname)
text_rep


# In[189]:


cloud_rep = WordCloud(background_color="white", colormap="Reds").generate(text_rep)
plt.imshow(cloud_rep, interpolation='bilinear') 
plt.axis("off")
# Don't forget to show the final image
plt.show()


# In[192]:


#adding a bit of fantasy by shaping the wordcloud
elephant = np.array(Image.open("republicans.jpg"))
cloud_shape_rep = WordCloud(background_color="white", mask=elephant).generate(text_rep)

image_colors_rep = ImageColorGenerator(elephant)
plt.figure(figsize=[7,7])
plt.imshow(cloud_shape_rep.recolor(color_func=image_colors_rep), interpolation="bilinear")
plt.axis("off")

# Don't forget to show the final image
plt.show()


# ## same for democrats:

# In[177]:


#creating a text block with all democrat names
text_dem = " ".join(df_democrats for df_democrats in df_democrats.surname)
cloud_dem = WordCloud(background_color="white", colormap="Blues").generate(text_dem)
plt.imshow(cloud_dem, interpolation='bilinear') 
plt.axis("off")
# Don't forget to show the final image
plt.show()


# In[194]:


#adding a bit of fantasy by shaping the wordcloud
donkey = np.array(Image.open("democrats.jpg"))
cloud_shape_dem = WordCloud(background_color="white", mask=donkey).generate(text_dem)

image_colors = ImageColorGenerator(donkey)
plt.figure(figsize=[7,7])
plt.imshow(cloud_shape_dem.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# Don't forget to show the final image
plt.show()


# ## Now let's play a bit with the wordcloud... I'd like to see the most popular names by state.

# In[156]:


california_mask = np.array(Image.open("california.jpg"))


# In[165]:


#most common names in California
df_cali = df.loc[df["state"]=="California"]
text_cali = " ".join(df_cali for df_cali in df_cali.surname)
cloud_cali = WordCloud(background_color="white", mask=california_mask).generate(text_cali)
plt.imshow(cloud_cali, interpolation='bilinear') 
plt.axis("off")
# Don't forget to show the final image
plt.show()


# In[168]:


get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[144]:


#define a function that creates a wordcloud of the most common names in {insert State}

def get_cloud(enter_state):

    df_state = df.loc[df["state"]==enter_state]
    text_state = " ".join(df_state for df_state in df_state.surname)
    cloud_state = WordCloud(background_color="white").generate(text_state)
    plt.imshow(cloud_state, interpolation='bilinear') 
    plt.axis("off")
    # Don't forget to show the final image
    plt.show()


# In[134]:


get_cloud("California")

