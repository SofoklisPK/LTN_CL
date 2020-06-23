import matplotlib.pyplot as plt
import csv
import pandas as pd


with open('axioms_values.csv', 'r') as f:
    df =  pd.read_csv(f)

df['sat_level'] = df.mean(numeric_only=True, axis=1)
df['sat_level'].plot()
plt.show()

df[['~Gray(object0)','~Blue(object0)','~Brown(object0)','~Yellow(object0)','~Red(object0)','~Green(object0)','Purple(object0)','~Cyan(object0)']].plot()
plt.show()

df[['Large(object0)','~Small(object0)']].plot()
plt.show()

df[['~Cube(object0)','~Sphere(object0)','Cylinder(object0)']].plot()
plt.show()

df[['Rubber(object0)','~Metal(object0)']].plot()
plt.show()

df[['Right(object0,object1)','~Left(object0,object1)']].plot()
plt.show()

df[['forall ?obj: Cyan(?obj) -> ~Purple(?obj)','forall ?obj: Small(?obj) -> ~Large(?obj)','forall ?obj: Cube(?obj) -> ~Sphere(?obj)','forall ?obj: Rubber(?obj) -> ~Metal(?obj)']].plot()
plt.show()

df[['forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Left(?obj, ?obj_2)','forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Right(?obj_2, ?obj)','forall ?obj: ~Right(?obj, ?obj)']].plot()
plt.show()