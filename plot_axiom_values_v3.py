import matplotlib.pyplot as plt
import csv
import pandas as pd


with open('axioms_values.csv', 'r') as f:
    df =  pd.read_csv(f)

df['sat_level'] = df.mean(numeric_only=True, axis=1)
df['sat_level'].plot()
plt.show()

df[['forall ?is_gray : Gray(?is_gray)','forall ?isnot_gray : ~Gray(?isnot_gray)']].plot()
plt.show()

df[['forall ?is_small : Small(?is_small)','forall ?isnot_small : ~Small(?isnot_small)','forall ?is_large : Large(?is_large)','forall ?isnot_large : ~Large(?isnot_large)']].plot()
plt.show()

df[['forall ?is_cube : Cube(?is_cube)','forall ?isnot_cube : ~Cube(?isnot_cube)']].plot()
plt.show()

df[['forall ?obj: Yellow(?obj) -> ~Red(?obj)','forall ?obj: Small(?obj) -> ~Large(?obj)','forall ?obj: Cube(?obj) -> ~Sphere(?obj)','forall ?obj: Rubber(?obj) -> ~Metal(?obj)']].plot()
plt.show()

#df[['forall ?right_pair : Right(?right_pair)','forall ?left_pair : ~Right(?left_pair)']].plot()
df[['forall ?right_pair : Right(?right_pair)','forall ?left_pair : ~Right(?left_pair)','forall ?obj: ~Behind(?obj, ?obj)']].plot()
plt.show()

