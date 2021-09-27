```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')
```


```python
url = 'https://raw.githubusercontent.com/umaimehm/Intro_to_AI_2021/main/assignment1/Ruter_data.csv'
```


```python
df = pd.read_csv(url, sep=';', error_bad_lines=False)
```


```python
#velger følgende features:  Kommune, Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra, 
#                          Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra, Kjøretøy_Kapasitet, Passasjerer_Ombord
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TurId</th>
      <th>Dato</th>
      <th>Fylke</th>
      <th>Område</th>
      <th>Kommune</th>
      <th>Holdeplass_Fra</th>
      <th>Holdeplass_Til</th>
      <th>Linjetype</th>
      <th>Linjefylke</th>
      <th>Linjenavn</th>
      <th>Linjeretning</th>
      <th>Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra</th>
      <th>Tidspunkt_Faktisk_Avgang_Holdeplass_Fra</th>
      <th>Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra</th>
      <th>Tidspunkt_Planlagt_Avgang_Holdeplass_Fra</th>
      <th>Kjøretøy_Kapasitet</th>
      <th>Passasjerer_Ombord</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15006-2020-08-10T10:24:00+02:00</td>
      <td>10/08/2020</td>
      <td>Viken</td>
      <td>Vest</td>
      <td>Bærum</td>
      <td>Nordliveien</td>
      <td>Tjernsmyr</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>150</td>
      <td>0</td>
      <td>10:53:53</td>
      <td>10:53:59</td>
      <td>10:53:00</td>
      <td>10:53:00</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15002-2020-08-15T12:54:00+02:00</td>
      <td>15/08/2020</td>
      <td>Viken</td>
      <td>Vest</td>
      <td>Bærum</td>
      <td>Nadderud stadion</td>
      <td>Bekkestua bussterminal (Plattform C)</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>150</td>
      <td>0</td>
      <td>13:12:20</td>
      <td>13:12:26</td>
      <td>13:12:00</td>
      <td>13:12:00</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15004-2020-08-03T09:54:00+02:00</td>
      <td>03/08/2020</td>
      <td>Viken</td>
      <td>Vest</td>
      <td>Bærum</td>
      <td>Ringstabekkveien</td>
      <td>Skallum</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>150</td>
      <td>0</td>
      <td>10:18:56</td>
      <td>10:19:21</td>
      <td>10:19:00</td>
      <td>10:19:00</td>
      <td>112</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15003-2020-07-27T13:00:00+02:00</td>
      <td>27/07/2020</td>
      <td>Viken</td>
      <td>Vest</td>
      <td>Bærum</td>
      <td>Gruvemyra</td>
      <td>Gullhaug</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>150</td>
      <td>1</td>
      <td>13:52:04</td>
      <td>13:52:26</td>
      <td>13:51:00</td>
      <td>13:51:00</td>
      <td>112</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15002-2020-08-27T07:15:00+02:00</td>
      <td>27/08/2020</td>
      <td>Viken</td>
      <td>Vest</td>
      <td>Bærum</td>
      <td>Lysaker stasjon (Plattform A)</td>
      <td>Tjernsmyr</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>150</td>
      <td>1</td>
      <td>07:34:13</td>
      <td>07:34:53</td>
      <td>07:33:00</td>
      <td>07:33:00</td>
      <td>112</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>10001-2020-06-10T15:10:00+02:00</td>
      <td>10/06/2020</td>
      <td>Viken</td>
      <td>Nordøst</td>
      <td>Lillestrøm</td>
      <td>Brauterkrysset</td>
      <td>Nordsnoveien</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>100</td>
      <td>1</td>
      <td>16:23:18</td>
      <td>16:23:48</td>
      <td>16:10:00</td>
      <td>16:10:00</td>
      <td>151</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>10010-2020-06-23T05:54:00+02:00</td>
      <td>23/06/2020</td>
      <td>Viken</td>
      <td>Nordøst</td>
      <td>Lillestrøm</td>
      <td>Vestbygata</td>
      <td>Bjørnsons gate</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>100</td>
      <td>0</td>
      <td>06:00:32</td>
      <td>06:00:40</td>
      <td>05:59:00</td>
      <td>05:59:00</td>
      <td>151</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>2007-2020-06-11T17:22:00+02:00</td>
      <td>11/06/2020</td>
      <td>Oslo</td>
      <td>Indre By</td>
      <td>Sagene</td>
      <td>Torshovparken  (mot Torshovparken)</td>
      <td>Torshov  (mot Bentsebrua)</td>
      <td>Lokal</td>
      <td>Oslo</td>
      <td>20</td>
      <td>1</td>
      <td>17:42:43</td>
      <td>17:43:10</td>
      <td>17:32:00</td>
      <td>17:32:00</td>
      <td>106</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>10004-2020-08-13T04:59:00+02:00</td>
      <td>13/08/2020</td>
      <td>Oslo</td>
      <td>Indre By</td>
      <td>Gamle Oslo</td>
      <td>Harald Hårdrådes plass  (mot Grønland)</td>
      <td>Oslo gate  (mot Grønland)</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>100</td>
      <td>0</td>
      <td>06:00:11</td>
      <td>06:00:23</td>
      <td>05:59:00</td>
      <td>05:59:00</td>
      <td>151</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>34013-2020-08-05T13:31:00+02:00</td>
      <td>05/08/2020</td>
      <td>Viken</td>
      <td>Nordøst</td>
      <td>Lillestrøm</td>
      <td>Tandberg</td>
      <td>Selmer</td>
      <td>Lokal</td>
      <td>Viken</td>
      <td>340</td>
      <td>1</td>
      <td>13:52:44</td>
      <td>13:52:48</td>
      <td>13:50:00</td>
      <td>13:50:00</td>
      <td>105</td>
      <td>-5</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 17 columns</p>
</div>




```python
df = df.drop(["TurId", "Dato", "Fylke", "Område", "Holdeplass_Fra", "Holdeplass_Til", "Linjetype", "Linjefylke", "Linjenavn", "Linjeretning", "Tidspunkt_Faktisk_Avgang_Holdeplass_Fra", "Tidspunkt_Planlagt_Avgang_Holdeplass_Fra"], axis=1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kommune</th>
      <th>Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra</th>
      <th>Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra</th>
      <th>Kjøretøy_Kapasitet</th>
      <th>Passasjerer_Ombord</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bærum</td>
      <td>10:53:53</td>
      <td>10:53:00</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bærum</td>
      <td>13:12:20</td>
      <td>13:12:00</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bærum</td>
      <td>10:18:56</td>
      <td>10:19:00</td>
      <td>112</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bærum</td>
      <td>13:52:04</td>
      <td>13:51:00</td>
      <td>112</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bærum</td>
      <td>07:34:13</td>
      <td>07:33:00</td>
      <td>112</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum()
```




    Kommune                                      0
    Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra     0
    Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra    0
    Kjøretøy_Kapasitet                           0
    Passasjerer_Ombord                           0
    dtype: int64




```python
df["Kommune"].value_counts()
```




    Lillestrøm           1094
    Bærum                 724
    Lørenskog             525
    Ullensaker            358
    Asker                 326
    Nittedal              274
    Gamle Oslo            262
    Rælingen              248
    Nannestad             236
    Alna                  210
    Bjerke                199
    Enebakk               153
    Eidsvoll              144
    Stovner               125
    Vestre Aker           121
    Nes                   117
    Grünerløkka           110
    Gjerdrum               97
    Aurskog-Høland         97
    Nordstrand             89
    Ullern                 85
    Grorud                 77
    Sentrum                77
    St.Hanshaugen          47
    Sagene                 43
    Frogner                33
    Søndre Nordstrand      30
    Nordre Follo           19
    Indre Østfold          18
    Nordre Aker            18
    Lier                   15
    Nordmarka              12
    Hurdal                 11
    Drammen                 6
    Name: Kommune, dtype: int64




```python
df["Kjøretøy_Kapasitet"].value_counts()
```




    106    1805
    112     791
    151     690
    115     492
    105     475
    80      452
    72      398
    130     205
    71      193
    69      174
    75       89
    33       68
    77       45
    76       45
    47       39
    70       34
    103       4
    102       1
    Name: Kjøretøy_Kapasitet, dtype: int64




```python
sns.countplot(x='Kommune', data=df);
plt.xticks(rotation=90);
```


    
![png](output_10_0.png)
    



```python
from datetime import datetime
df['Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra'] = pd.to_datetime(df['Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra'], errors='coerce')
df['Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra'] = pd.to_datetime(df['Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra'], errors='coerce')
df['forsinkelse'] = df["Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra"]-df["Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra"]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kommune</th>
      <th>Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra</th>
      <th>Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra</th>
      <th>Kjøretøy_Kapasitet</th>
      <th>Passasjerer_Ombord</th>
      <th>forsinkelse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bærum</td>
      <td>2021-09-27 10:53:53</td>
      <td>2021-09-27 10:53:00</td>
      <td>112</td>
      <td>5</td>
      <td>0 days 00:00:53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bærum</td>
      <td>2021-09-27 13:12:20</td>
      <td>2021-09-27 13:12:00</td>
      <td>112</td>
      <td>5</td>
      <td>0 days 00:00:20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bærum</td>
      <td>2021-09-27 10:18:56</td>
      <td>2021-09-27 10:19:00</td>
      <td>112</td>
      <td>6</td>
      <td>-1 days +23:59:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bærum</td>
      <td>2021-09-27 13:52:04</td>
      <td>2021-09-27 13:51:00</td>
      <td>112</td>
      <td>10</td>
      <td>0 days 00:01:04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bærum</td>
      <td>2021-09-27 07:34:13</td>
      <td>2021-09-27 07:33:00</td>
      <td>112</td>
      <td>10</td>
      <td>0 days 00:01:13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>Lillestrøm</td>
      <td>2021-09-27 16:23:18</td>
      <td>2021-09-27 16:10:00</td>
      <td>151</td>
      <td>2</td>
      <td>0 days 00:13:18</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>Lillestrøm</td>
      <td>2021-09-27 06:00:32</td>
      <td>2021-09-27 05:59:00</td>
      <td>151</td>
      <td>2</td>
      <td>0 days 00:01:32</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>Sagene</td>
      <td>2021-09-27 17:42:43</td>
      <td>2021-09-27 17:32:00</td>
      <td>106</td>
      <td>3</td>
      <td>0 days 00:10:43</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>Gamle Oslo</td>
      <td>2021-09-27 06:00:11</td>
      <td>2021-09-27 05:59:00</td>
      <td>151</td>
      <td>5</td>
      <td>0 days 00:01:11</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>Lillestrøm</td>
      <td>2021-09-27 13:52:44</td>
      <td>2021-09-27 13:50:00</td>
      <td>105</td>
      <td>-5</td>
      <td>0 days 00:02:44</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 6 columns</p>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kommune</th>
      <th>Tidspunkt_Faktisk_Ankomst_Holdeplass_Fra</th>
      <th>Tidspunkt_Planlagt_Ankomst_Holdeplass_Fra</th>
      <th>Kjøretøy_Kapasitet</th>
      <th>Passasjerer_Ombord</th>
      <th>forsinkelse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bærum</td>
      <td>2021-09-27 10:53:53</td>
      <td>2021-09-27 10:53:00</td>
      <td>112</td>
      <td>5</td>
      <td>0 days 00:00:53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bærum</td>
      <td>2021-09-27 13:12:20</td>
      <td>2021-09-27 13:12:00</td>
      <td>112</td>
      <td>5</td>
      <td>0 days 00:00:20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bærum</td>
      <td>2021-09-27 10:18:56</td>
      <td>2021-09-27 10:19:00</td>
      <td>112</td>
      <td>6</td>
      <td>-1 days +23:59:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bærum</td>
      <td>2021-09-27 13:52:04</td>
      <td>2021-09-27 13:51:00</td>
      <td>112</td>
      <td>10</td>
      <td>0 days 00:01:04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bærum</td>
      <td>2021-09-27 07:34:13</td>
      <td>2021-09-27 07:33:00</td>
      <td>112</td>
      <td>10</td>
      <td>0 days 00:01:13</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['forsinkelse'].describe()
```




    count                         5775
    mean     0 days 00:01:33.185800865
    std      0 days 00:05:29.733101974
    min              -1 days +23:16:55
    25%                0 days 00:00:30
    50%                0 days 00:01:25
    75%                0 days 00:02:39
    max                0 days 02:33:31
    Name: forsinkelse, dtype: object




```python
df['prosent_opptatt'] = (100* df['Passasjerer_Ombord']) / df['Kjøretøy_Kapasitet']
```


```python
df['prosent_opptatt'].describe()
```




    count    6000.000000
    mean        4.396286
    std         6.825213
    min       -34.821429
    25%         0.000000
    50%         2.830189
    75%         7.142857
    max        80.000000
    Name: prosent_opptatt, dtype: float64




```python
df['prosent_opptatt'].plot.box()
```




    <AxesSubplot:>




    
![png](output_17_1.png)
    



```python

```
