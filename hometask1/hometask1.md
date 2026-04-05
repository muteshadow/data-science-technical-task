<p align="center"><img src="../assets/img/amazinum.png" width="180"></p>

# Hometask: pandas

## Question 1


```python
import pandas as pd
import numpy as np
```


```python
energy = pd.read_excel('data/Energy Indicators.xls', skiprows=17, skipfooter=38)

# delete first two columns
energy = energy.iloc[:, 2:]
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

# missing data reflect as nan
energy = energy.replace('...', np.nan)

# convert Energy Supply to gigajoules
energy['Energy Supply'] = energy['Energy Supply'] * 1000000 # 1 000 000

# видаляє дужки і їх вміст
energy['Country'] = energy['Country'].str.replace(r"\(.*\)", "", regex=True)
# видаляє всі цифри
energy['Country'] = energy['Country'].str.replace(r"\d+", "", regex=True)
# прибирає пробіли
energy['Country'] = energy['Country'].str.strip()

energy = energy.replace({
    "Republic of Korea": "South Korea",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "China, Hong Kong Special Administrative Region": "Hong Kong"
})

# чомусь сприймає як об'єкти
energy['Energy Supply'] = pd.to_numeric(energy['Energy Supply'])
energy['Energy Supply per Capita'] = pd.to_numeric(energy['Energy Supply per Capita'])

energy.head()

# перевірка типів
# energy.info()

# адекватність назв країн
# list(energy['Country'].unique())
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Energy Supply</th>
      <th>Energy Supply per Capita</th>
      <th>% Renewable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>3.210000e+08</td>
      <td>10.0</td>
      <td>78.669280</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>1.020000e+08</td>
      <td>35.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>1.959000e+09</td>
      <td>51.0</td>
      <td>0.551010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>American Samoa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.641026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>9.000000e+06</td>
      <td>121.0</td>
      <td>88.695650</td>
    </tr>
  </tbody>
</table>


```python
GPD = pd.read_excel('data/world_bank.xls', skiprows=3)

GPD = GPD.rename(columns={"Country Name": "Country"})

GPD = GPD.replace({
    "Korea, Rep.": "South Korea",
    "Iran, Islamic Rep.": "Iran",
    "Hong Kong SAR, China": "Hong Kong"
})

GPD.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Country Code</th>
      <th>Indicator Name</th>
      <th>Indicator Code</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>...</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>GDP (current US$)</td>
      <td>NY.GDP.MKTP.CD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.727933e+09</td>
      <td>2.791061e+09</td>
      <td>2.963128e+09</td>
      <td>2.983799e+09</td>
      <td>3.092179e+09</td>
      <td>3.276188e+09</td>
      <td>3.395794e+09</td>
      <td>2.610039e+09</td>
      <td>3.126019e+09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Africa Eastern and Southern</td>
      <td>AFE</td>
      <td>GDP (current US$)</td>
      <td>NY.GDP.MKTP.CD</td>
      <td>2.112502e+10</td>
      <td>2.161623e+10</td>
      <td>2.350628e+10</td>
      <td>2.804836e+10</td>
      <td>2.592067e+10</td>
      <td>2.947210e+10</td>
      <td>...</td>
      <td>9.859871e+11</td>
      <td>1.006526e+12</td>
      <td>9.273485e+11</td>
      <td>8.851764e+11</td>
      <td>1.021043e+12</td>
      <td>1.007196e+12</td>
      <td>1.000834e+12</td>
      <td>9.275933e+11</td>
      <td>1.081998e+12</td>
      <td>1.169484e+12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>GDP (current US$)</td>
      <td>NY.GDP.MKTP.CD</td>
      <td>5.377778e+08</td>
      <td>5.488889e+08</td>
      <td>5.466667e+08</td>
      <td>7.511112e+08</td>
      <td>8.000000e+08</td>
      <td>1.006667e+09</td>
      <td>...</td>
      <td>2.056449e+10</td>
      <td>2.055058e+10</td>
      <td>1.999814e+10</td>
      <td>1.801955e+10</td>
      <td>1.889635e+10</td>
      <td>1.841886e+10</td>
      <td>1.890450e+10</td>
      <td>2.014345e+10</td>
      <td>1.458314e+10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Africa Western and Central</td>
      <td>AFW</td>
      <td>GDP (current US$)</td>
      <td>NY.GDP.MKTP.CD</td>
      <td>1.044764e+10</td>
      <td>1.117321e+10</td>
      <td>1.199053e+10</td>
      <td>1.272769e+10</td>
      <td>1.389811e+10</td>
      <td>1.492979e+10</td>
      <td>...</td>
      <td>8.339481e+11</td>
      <td>8.943225e+11</td>
      <td>7.686447e+11</td>
      <td>6.913634e+11</td>
      <td>6.848988e+11</td>
      <td>7.670257e+11</td>
      <td>8.225384e+11</td>
      <td>7.864600e+11</td>
      <td>8.444597e+11</td>
      <td>8.778633e+11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>AGO</td>
      <td>GDP (current US$)</td>
      <td>NY.GDP.MKTP.CD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.334016e+11</td>
      <td>1.372444e+11</td>
      <td>8.721930e+10</td>
      <td>4.984049e+10</td>
      <td>6.897277e+10</td>
      <td>7.779294e+10</td>
      <td>6.930911e+10</td>
      <td>5.024137e+10</td>
      <td>6.568544e+10</td>
      <td>1.067136e+11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 67 columns</p>


```python
ScimEn = pd.read_excel('data/scimagojr.xlsx')

ScimEn.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Country</th>
      <th>Region</th>
      <th>Documents</th>
      <th>Citable documents</th>
      <th>Citations</th>
      <th>Self-citations</th>
      <th>Citations per document</th>
      <th>H index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>China</td>
      <td>Asiatic Region</td>
      <td>273437</td>
      <td>272374</td>
      <td>2336764</td>
      <td>1615239</td>
      <td>8.55</td>
      <td>245</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>United States</td>
      <td>Northern America</td>
      <td>175891</td>
      <td>172431</td>
      <td>2230544</td>
      <td>724472</td>
      <td>12.68</td>
      <td>363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>India</td>
      <td>Asiatic Region</td>
      <td>55082</td>
      <td>53775</td>
      <td>463165</td>
      <td>162944</td>
      <td>8.41</td>
      <td>181</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Japan</td>
      <td>Asiatic Region</td>
      <td>50523</td>
      <td>50065</td>
      <td>488062</td>
      <td>119930</td>
      <td>9.66</td>
      <td>193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>United Kingdom</td>
      <td>Western Europe</td>
      <td>43389</td>
      <td>42284</td>
      <td>615670</td>
      <td>111290</td>
      <td>14.19</td>
      <td>226</td>
    </tr>
  </tbody>
</table>


```python
# only the top 15 countries
ScimEn = ScimEn[ScimEn['Rank'] <= 15]
                
# inner - тільки перетин
df = pd.merge(ScimEn, energy, how='inner', on='Country')
df = pd.merge(df, GPD, how='inner', on='Country')

# only the 10 years (2006-2015) of GDP data
years = [str(year) for year in range (2006, 2016)]

columns = ['Rank', 'Documents', 'Citable documents', 'Citations',
           'Self-citations', 'Citations per document', 'H index',
           'Energy Supply', 'Energy Supply per Capita', '% Renewable'] + years

df = df.set_index('Country')
df = df[columns]

# розмір таблиці (рядки х стовпці)
# df.shape
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Documents</th>
      <th>Citable documents</th>
      <th>Citations</th>
      <th>Self-citations</th>
      <th>Citations per document</th>
      <th>H index</th>
      <th>Energy Supply</th>
      <th>Energy Supply per Capita</th>
      <th>% Renewable</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>1</td>
      <td>273437</td>
      <td>272374</td>
      <td>2336764</td>
      <td>1615239</td>
      <td>8.55</td>
      <td>245</td>
      <td>1.271910e+11</td>
      <td>93.0</td>
      <td>19.754910</td>
      <td>2.752119e+12</td>
      <td>3.550328e+12</td>
      <td>4.594337e+12</td>
      <td>5.101691e+12</td>
      <td>6.087192e+12</td>
      <td>7.551545e+12</td>
      <td>8.532186e+12</td>
      <td>9.570471e+12</td>
      <td>1.047562e+13</td>
      <td>1.106157e+13</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>2</td>
      <td>175891</td>
      <td>172431</td>
      <td>2230544</td>
      <td>724472</td>
      <td>12.68</td>
      <td>363</td>
      <td>9.083800e+10</td>
      <td>286.0</td>
      <td>11.570980</td>
      <td>1.381559e+13</td>
      <td>1.447423e+13</td>
      <td>1.476986e+13</td>
      <td>1.447806e+13</td>
      <td>1.504896e+13</td>
      <td>1.559973e+13</td>
      <td>1.625397e+13</td>
      <td>1.684319e+13</td>
      <td>1.755068e+13</td>
      <td>1.820602e+13</td>
    </tr>
    <tr>
      <th>India</th>
      <td>3</td>
      <td>55082</td>
      <td>53775</td>
      <td>463165</td>
      <td>162944</td>
      <td>8.41</td>
      <td>181</td>
      <td>3.319500e+10</td>
      <td>26.0</td>
      <td>14.969080</td>
      <td>9.402599e+11</td>
      <td>1.216736e+12</td>
      <td>1.198895e+12</td>
      <td>1.341888e+12</td>
      <td>1.675616e+12</td>
      <td>1.823052e+12</td>
      <td>1.827638e+12</td>
      <td>1.856721e+12</td>
      <td>2.039126e+12</td>
      <td>2.103588e+12</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4</td>
      <td>50523</td>
      <td>50065</td>
      <td>488062</td>
      <td>119930</td>
      <td>9.66</td>
      <td>193</td>
      <td>1.898400e+10</td>
      <td>149.0</td>
      <td>10.232820</td>
      <td>4.601663e+12</td>
      <td>4.579750e+12</td>
      <td>5.106679e+12</td>
      <td>5.289494e+12</td>
      <td>5.759072e+12</td>
      <td>6.233147e+12</td>
      <td>6.272363e+12</td>
      <td>5.212328e+12</td>
      <td>4.896994e+12</td>
      <td>4.444931e+12</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>5</td>
      <td>43389</td>
      <td>42284</td>
      <td>615670</td>
      <td>111290</td>
      <td>14.19</td>
      <td>226</td>
      <td>7.920000e+09</td>
      <td>124.0</td>
      <td>10.600470</td>
      <td>2.709978e+12</td>
      <td>3.092996e+12</td>
      <td>2.931684e+12</td>
      <td>2.417566e+12</td>
      <td>2.491397e+12</td>
      <td>2.666403e+12</td>
      <td>2.706341e+12</td>
      <td>2.786315e+12</td>
      <td>3.065223e+12</td>
      <td>2.934858e+12</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>6</td>
      <td>38739</td>
      <td>38013</td>
      <td>433148</td>
      <td>95145</td>
      <td>11.18</td>
      <td>196</td>
      <td>1.326100e+10</td>
      <td>165.0</td>
      <td>17.901530</td>
      <td>2.994704e+12</td>
      <td>3.425578e+12</td>
      <td>3.745264e+12</td>
      <td>3.411261e+12</td>
      <td>3.399668e+12</td>
      <td>3.749315e+12</td>
      <td>3.527143e+12</td>
      <td>3.733805e+12</td>
      <td>3.889093e+12</td>
      <td>3.357586e+12</td>
    </tr>
    <tr>
      <th>Russian Federation</th>
      <td>7</td>
      <td>36735</td>
      <td>36560</td>
      <td>115938</td>
      <td>54993</td>
      <td>3.16</td>
      <td>90</td>
      <td>3.070900e+10</td>
      <td>214.0</td>
      <td>17.288680</td>
      <td>9.899321e+11</td>
      <td>1.299703e+12</td>
      <td>1.660848e+12</td>
      <td>1.222646e+12</td>
      <td>1.524917e+12</td>
      <td>2.045923e+12</td>
      <td>2.208294e+12</td>
      <td>2.292470e+12</td>
      <td>2.059242e+12</td>
      <td>1.363482e+12</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>8</td>
      <td>33472</td>
      <td>32863</td>
      <td>568080</td>
      <td>100953</td>
      <td>16.97</td>
      <td>227</td>
      <td>1.043100e+10</td>
      <td>296.0</td>
      <td>61.945430</td>
      <td>1.319265e+12</td>
      <td>1.468820e+12</td>
      <td>1.552990e+12</td>
      <td>1.374625e+12</td>
      <td>1.617343e+12</td>
      <td>1.793327e+12</td>
      <td>1.828366e+12</td>
      <td>1.846597e+12</td>
      <td>1.805750e+12</td>
      <td>1.556509e+12</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>9</td>
      <td>27983</td>
      <td>26940</td>
      <td>352993</td>
      <td>87828</td>
      <td>12.61</td>
      <td>166</td>
      <td>6.530000e+09</td>
      <td>109.0</td>
      <td>33.667230</td>
      <td>1.949552e+12</td>
      <td>2.213102e+12</td>
      <td>2.408655e+12</td>
      <td>2.199929e+12</td>
      <td>2.136100e+12</td>
      <td>2.294994e+12</td>
      <td>2.086958e+12</td>
      <td>2.141924e+12</td>
      <td>2.162010e+12</td>
      <td>1.836638e+12</td>
    </tr>
    <tr>
      <th>South Korea</th>
      <td>10</td>
      <td>27655</td>
      <td>27445</td>
      <td>328488</td>
      <td>61531</td>
      <td>11.88</td>
      <td>155</td>
      <td>1.100700e+10</td>
      <td>221.0</td>
      <td>2.279353</td>
      <td>1.053217e+12</td>
      <td>1.172614e+12</td>
      <td>1.047339e+12</td>
      <td>9.439419e+11</td>
      <td>1.144067e+12</td>
      <td>1.253223e+12</td>
      <td>1.278428e+12</td>
      <td>1.370795e+12</td>
      <td>1.484318e+12</td>
      <td>1.465773e+12</td>
    </tr>
    <tr>
      <th>France</th>
      <td>11</td>
      <td>25232</td>
      <td>24732</td>
      <td>343860</td>
      <td>65734</td>
      <td>13.63</td>
      <td>178</td>
      <td>1.059700e+10</td>
      <td>166.0</td>
      <td>17.020280</td>
      <td>2.320536e+12</td>
      <td>2.660591e+12</td>
      <td>2.930304e+12</td>
      <td>2.700887e+12</td>
      <td>2.645188e+12</td>
      <td>2.865158e+12</td>
      <td>2.683672e+12</td>
      <td>2.811877e+12</td>
      <td>2.855964e+12</td>
      <td>2.439189e+12</td>
    </tr>
    <tr>
      <th>Iran</th>
      <td>12</td>
      <td>22933</td>
      <td>22734</td>
      <td>307280</td>
      <td>97038</td>
      <td>13.40</td>
      <td>141</td>
      <td>9.172000e+09</td>
      <td>119.0</td>
      <td>5.707721</td>
      <td>2.662989e+11</td>
      <td>3.498816e+11</td>
      <td>4.123362e+11</td>
      <td>4.163970e+11</td>
      <td>4.868076e+11</td>
      <td>6.261331e+11</td>
      <td>6.440355e+11</td>
      <td>4.927756e+11</td>
      <td>4.603828e+11</td>
      <td>4.082129e+11</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>13</td>
      <td>21955</td>
      <td>21597</td>
      <td>352497</td>
      <td>64588</td>
      <td>16.06</td>
      <td>176</td>
      <td>4.923000e+09</td>
      <td>106.0</td>
      <td>37.968590</td>
      <td>1.260399e+12</td>
      <td>1.474003e+12</td>
      <td>1.631863e+12</td>
      <td>1.491473e+12</td>
      <td>1.422108e+12</td>
      <td>1.480710e+12</td>
      <td>1.324751e+12</td>
      <td>1.355580e+12</td>
      <td>1.371821e+12</td>
      <td>1.196157e+12</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>14</td>
      <td>21524</td>
      <td>21236</td>
      <td>183915</td>
      <td>45172</td>
      <td>8.54</td>
      <td>127</td>
      <td>1.214900e+10</td>
      <td>59.0</td>
      <td>69.648030</td>
      <td>1.107627e+12</td>
      <td>1.397114e+12</td>
      <td>1.695855e+12</td>
      <td>1.666996e+12</td>
      <td>2.208838e+12</td>
      <td>2.616157e+12</td>
      <td>2.465228e+12</td>
      <td>2.472819e+12</td>
      <td>2.456044e+12</td>
      <td>1.802212e+12</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>15</td>
      <td>20614</td>
      <td>20147</td>
      <td>314307</td>
      <td>51583</td>
      <td>15.25</td>
      <td>176</td>
      <td>5.386000e+09</td>
      <td>231.0</td>
      <td>11.810810</td>
      <td>7.479074e+11</td>
      <td>8.544273e+11</td>
      <td>1.055686e+12</td>
      <td>9.286298e+11</td>
      <td>1.148570e+12</td>
      <td>1.398456e+12</td>
      <td>1.546953e+12</td>
      <td>1.576330e+12</td>
      <td>1.467590e+12</td>
      <td>1.350580e+12</td>
    </tr>
  </tbody>
</table>


### Function "answer_one" return the resulted DataFrame


```python
def answer_one():
    return df

answer_one()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Documents</th>
      <th>Citable documents</th>
      <th>Citations</th>
      <th>Self-citations</th>
      <th>Citations per document</th>
      <th>H index</th>
      <th>Energy Supply</th>
      <th>Energy Supply per Capita</th>
      <th>% Renewable</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>1</td>
      <td>273437</td>
      <td>272374</td>
      <td>2336764</td>
      <td>1615239</td>
      <td>8.55</td>
      <td>245</td>
      <td>1.271910e+11</td>
      <td>93.0</td>
      <td>19.754910</td>
      <td>2.752119e+12</td>
      <td>3.550328e+12</td>
      <td>4.594337e+12</td>
      <td>5.101691e+12</td>
      <td>6.087192e+12</td>
      <td>7.551545e+12</td>
      <td>8.532186e+12</td>
      <td>9.570471e+12</td>
      <td>1.047562e+13</td>
      <td>1.106157e+13</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>2</td>
      <td>175891</td>
      <td>172431</td>
      <td>2230544</td>
      <td>724472</td>
      <td>12.68</td>
      <td>363</td>
      <td>9.083800e+10</td>
      <td>286.0</td>
      <td>11.570980</td>
      <td>1.381559e+13</td>
      <td>1.447423e+13</td>
      <td>1.476986e+13</td>
      <td>1.447806e+13</td>
      <td>1.504896e+13</td>
      <td>1.559973e+13</td>
      <td>1.625397e+13</td>
      <td>1.684319e+13</td>
      <td>1.755068e+13</td>
      <td>1.820602e+13</td>
    </tr>
    <tr>
      <th>India</th>
      <td>3</td>
      <td>55082</td>
      <td>53775</td>
      <td>463165</td>
      <td>162944</td>
      <td>8.41</td>
      <td>181</td>
      <td>3.319500e+10</td>
      <td>26.0</td>
      <td>14.969080</td>
      <td>9.402599e+11</td>
      <td>1.216736e+12</td>
      <td>1.198895e+12</td>
      <td>1.341888e+12</td>
      <td>1.675616e+12</td>
      <td>1.823052e+12</td>
      <td>1.827638e+12</td>
      <td>1.856721e+12</td>
      <td>2.039126e+12</td>
      <td>2.103588e+12</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>4</td>
      <td>50523</td>
      <td>50065</td>
      <td>488062</td>
      <td>119930</td>
      <td>9.66</td>
      <td>193</td>
      <td>1.898400e+10</td>
      <td>149.0</td>
      <td>10.232820</td>
      <td>4.601663e+12</td>
      <td>4.579750e+12</td>
      <td>5.106679e+12</td>
      <td>5.289494e+12</td>
      <td>5.759072e+12</td>
      <td>6.233147e+12</td>
      <td>6.272363e+12</td>
      <td>5.212328e+12</td>
      <td>4.896994e+12</td>
      <td>4.444931e+12</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>5</td>
      <td>43389</td>
      <td>42284</td>
      <td>615670</td>
      <td>111290</td>
      <td>14.19</td>
      <td>226</td>
      <td>7.920000e+09</td>
      <td>124.0</td>
      <td>10.600470</td>
      <td>2.709978e+12</td>
      <td>3.092996e+12</td>
      <td>2.931684e+12</td>
      <td>2.417566e+12</td>
      <td>2.491397e+12</td>
      <td>2.666403e+12</td>
      <td>2.706341e+12</td>
      <td>2.786315e+12</td>
      <td>3.065223e+12</td>
      <td>2.934858e+12</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>6</td>
      <td>38739</td>
      <td>38013</td>
      <td>433148</td>
      <td>95145</td>
      <td>11.18</td>
      <td>196</td>
      <td>1.326100e+10</td>
      <td>165.0</td>
      <td>17.901530</td>
      <td>2.994704e+12</td>
      <td>3.425578e+12</td>
      <td>3.745264e+12</td>
      <td>3.411261e+12</td>
      <td>3.399668e+12</td>
      <td>3.749315e+12</td>
      <td>3.527143e+12</td>
      <td>3.733805e+12</td>
      <td>3.889093e+12</td>
      <td>3.357586e+12</td>
    </tr>
    <tr>
      <th>Russian Federation</th>
      <td>7</td>
      <td>36735</td>
      <td>36560</td>
      <td>115938</td>
      <td>54993</td>
      <td>3.16</td>
      <td>90</td>
      <td>3.070900e+10</td>
      <td>214.0</td>
      <td>17.288680</td>
      <td>9.899321e+11</td>
      <td>1.299703e+12</td>
      <td>1.660848e+12</td>
      <td>1.222646e+12</td>
      <td>1.524917e+12</td>
      <td>2.045923e+12</td>
      <td>2.208294e+12</td>
      <td>2.292470e+12</td>
      <td>2.059242e+12</td>
      <td>1.363482e+12</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>8</td>
      <td>33472</td>
      <td>32863</td>
      <td>568080</td>
      <td>100953</td>
      <td>16.97</td>
      <td>227</td>
      <td>1.043100e+10</td>
      <td>296.0</td>
      <td>61.945430</td>
      <td>1.319265e+12</td>
      <td>1.468820e+12</td>
      <td>1.552990e+12</td>
      <td>1.374625e+12</td>
      <td>1.617343e+12</td>
      <td>1.793327e+12</td>
      <td>1.828366e+12</td>
      <td>1.846597e+12</td>
      <td>1.805750e+12</td>
      <td>1.556509e+12</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>9</td>
      <td>27983</td>
      <td>26940</td>
      <td>352993</td>
      <td>87828</td>
      <td>12.61</td>
      <td>166</td>
      <td>6.530000e+09</td>
      <td>109.0</td>
      <td>33.667230</td>
      <td>1.949552e+12</td>
      <td>2.213102e+12</td>
      <td>2.408655e+12</td>
      <td>2.199929e+12</td>
      <td>2.136100e+12</td>
      <td>2.294994e+12</td>
      <td>2.086958e+12</td>
      <td>2.141924e+12</td>
      <td>2.162010e+12</td>
      <td>1.836638e+12</td>
    </tr>
    <tr>
      <th>South Korea</th>
      <td>10</td>
      <td>27655</td>
      <td>27445</td>
      <td>328488</td>
      <td>61531</td>
      <td>11.88</td>
      <td>155</td>
      <td>1.100700e+10</td>
      <td>221.0</td>
      <td>2.279353</td>
      <td>1.053217e+12</td>
      <td>1.172614e+12</td>
      <td>1.047339e+12</td>
      <td>9.439419e+11</td>
      <td>1.144067e+12</td>
      <td>1.253223e+12</td>
      <td>1.278428e+12</td>
      <td>1.370795e+12</td>
      <td>1.484318e+12</td>
      <td>1.465773e+12</td>
    </tr>
    <tr>
      <th>France</th>
      <td>11</td>
      <td>25232</td>
      <td>24732</td>
      <td>343860</td>
      <td>65734</td>
      <td>13.63</td>
      <td>178</td>
      <td>1.059700e+10</td>
      <td>166.0</td>
      <td>17.020280</td>
      <td>2.320536e+12</td>
      <td>2.660591e+12</td>
      <td>2.930304e+12</td>
      <td>2.700887e+12</td>
      <td>2.645188e+12</td>
      <td>2.865158e+12</td>
      <td>2.683672e+12</td>
      <td>2.811877e+12</td>
      <td>2.855964e+12</td>
      <td>2.439189e+12</td>
    </tr>
    <tr>
      <th>Iran</th>
      <td>12</td>
      <td>22933</td>
      <td>22734</td>
      <td>307280</td>
      <td>97038</td>
      <td>13.40</td>
      <td>141</td>
      <td>9.172000e+09</td>
      <td>119.0</td>
      <td>5.707721</td>
      <td>2.662989e+11</td>
      <td>3.498816e+11</td>
      <td>4.123362e+11</td>
      <td>4.163970e+11</td>
      <td>4.868076e+11</td>
      <td>6.261331e+11</td>
      <td>6.440355e+11</td>
      <td>4.927756e+11</td>
      <td>4.603828e+11</td>
      <td>4.082129e+11</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>13</td>
      <td>21955</td>
      <td>21597</td>
      <td>352497</td>
      <td>64588</td>
      <td>16.06</td>
      <td>176</td>
      <td>4.923000e+09</td>
      <td>106.0</td>
      <td>37.968590</td>
      <td>1.260399e+12</td>
      <td>1.474003e+12</td>
      <td>1.631863e+12</td>
      <td>1.491473e+12</td>
      <td>1.422108e+12</td>
      <td>1.480710e+12</td>
      <td>1.324751e+12</td>
      <td>1.355580e+12</td>
      <td>1.371821e+12</td>
      <td>1.196157e+12</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>14</td>
      <td>21524</td>
      <td>21236</td>
      <td>183915</td>
      <td>45172</td>
      <td>8.54</td>
      <td>127</td>
      <td>1.214900e+10</td>
      <td>59.0</td>
      <td>69.648030</td>
      <td>1.107627e+12</td>
      <td>1.397114e+12</td>
      <td>1.695855e+12</td>
      <td>1.666996e+12</td>
      <td>2.208838e+12</td>
      <td>2.616157e+12</td>
      <td>2.465228e+12</td>
      <td>2.472819e+12</td>
      <td>2.456044e+12</td>
      <td>1.802212e+12</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>15</td>
      <td>20614</td>
      <td>20147</td>
      <td>314307</td>
      <td>51583</td>
      <td>15.25</td>
      <td>176</td>
      <td>5.386000e+09</td>
      <td>231.0</td>
      <td>11.810810</td>
      <td>7.479074e+11</td>
      <td>8.544273e+11</td>
      <td>1.055686e+12</td>
      <td>9.286298e+11</td>
      <td>1.148570e+12</td>
      <td>1.398456e+12</td>
      <td>1.546953e+12</td>
      <td>1.576330e+12</td>
      <td>1.467590e+12</td>
      <td>1.350580e+12</td>
    </tr>
  </tbody>
</table>


## Question 2

### Function "avgGPD" return 15 countries and their average GDP sorted DESC


```python
def answer_two():
    Top15 = answer_one()

    years = [str(year) for year in range(2006, 2016)]

    avgGPD = Top15[years].mean(axis=1)
    avgGPD = avgGPD.sort_values(ascending=False)

    return avgGPD

answer_two()
```


    Country
    United States         1.570403e+13
    China                 6.927707e+12
    Japan                 5.239642e+12
    Germany               3.523342e+12
    United Kingdom        2.780276e+12
    France                2.691337e+12
    Italy                 2.142986e+12
    Brazil                1.988889e+12
    Russian Federation    1.666746e+12
    Canada                1.616359e+12
    India                 1.602352e+12
    Spain                 1.400886e+12
    South Korea           1.221372e+12
    Australia             1.207513e+12
    Iran                  4.563261e+11
    dtype: float64


## Question 3

### Function return how much had the GDP changed


```python
def answer_three():
    Top15 = answer_one()

    years = [str(year) for year in range(2006, 2016)]

    avgGPD = Top15[years].mean(axis=1)
    country = avgGPD.sort_values(ascending=False).index[5]

    change = Top15.loc[country, '2015'] - Top15.loc[country, '2006']

    return change
    # тільки результат
    # return float(change) 

answer_three()
```


    np.float64(118652421857.7959)


**manual result check**


```python
Top15 = answer_one()
Top15.loc['France', ['2006', '2015']]
```


    2006    2.320536e+12
    2015    2.439189e+12
    Name: France, dtype: float64


## Question 4

### Function return Self-Citations to Total Citations


```python
def answer_four():
    Top15 = answer_one()

    Top15['Ratio'] = Top15['Self-citations'] / Top15['Citations']

    max_country = Top15['Ratio'].idxmax()
    max_value = Top15['Ratio'].max()

    return(max_country, max_value)

answer_four()
```


    ('China', np.float64(0.6912289816173135))


## Question 5

### Function return the third most populous country


```python
def answer_five():
    Top15 = answer_one()

    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    country = Top15['Population'].sort_values(ascending=False).index[2]

    return country

answer_five()
```


    'United States'


## Question 6

### Function return correlation


```python
def answer_six():
    Top15 = answer_one()

    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['Population']

    correlation = Top15['Citable docs per Capita'].corr(Top15['Energy Supply per Capita'])

    return correlation

answer_six()
```


    np.float64(0.7434709127726777)


## Question 7

### Function return group the Countries by Continent


```python
def answer_seven():
    Top15 = answer_one()

    ContinentDict = {'China':'Asia',
                     'United States':'North America', 
                     'Japan':'Asia', 
                     'United Kingdom':'Europe', 
                     'Russian Federation':'Europe', 
                     'Canada':'North America', 
                     'Germany':'Europe', 
                     'India':'Asia',
                     'France':'Europe', 
                     'South Korea':'Asia', 
                     'Italy':'Europe', 
                     'Spain':'Europe', 
                     'Iran':'Asia',
                     'Australia':'Australia', 
                     'Brazil':'South America'}

    Top15['Continent'] = Top15.index.map(ContinentDict)
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    result = Top15.groupby('Continent')['Population'].agg(['size', 'sum', 'mean', 'std'])
    # size - кількість країн, sum - загальне населення
    # mean - середнє населення, std - різниця населення між країнами

    return result

answer_seven()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asia</th>
      <td>5</td>
      <td>2.898666e+09</td>
      <td>5.797333e+08</td>
      <td>6.790979e+08</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1</td>
      <td>2.331602e+07</td>
      <td>2.331602e+07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>6</td>
      <td>4.579297e+08</td>
      <td>7.632161e+07</td>
      <td>3.464767e+07</td>
    </tr>
    <tr>
      <th>North America</th>
      <td>2</td>
      <td>3.528552e+08</td>
      <td>1.764276e+08</td>
      <td>1.996696e+08</td>
    </tr>
    <tr>
      <th>South America</th>
      <td>1</td>
      <td>2.059153e+08</td>
      <td>2.059153e+08</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
