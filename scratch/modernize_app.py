with open(r'C:\Coding_Crusader\NEXUS\app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# In Streamlit 1.65, plotly_chart and dataframe can take width='stretch' or width=None
text = text.replace('use_container_width=True', "width='stretch'")
text = text.replace('use_container_width=False', "width='content'")

with open(r'C:\Coding_Crusader\NEXUS\app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('app.py successfully modernized for Streamlit 1.65+')
