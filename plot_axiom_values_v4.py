import matplotlib.pyplot as plt
import csv
import pandas as pd


with open('axioms_values_v4_with_equalities.csv', 'r') as f:
    df =  pd.read_csv(f)

df[['Green(object0)','Small(object0)','Sphere(object0)','Metal(object0)']].plot()
plt.show()

df[['forall ?obj: Blue(?obj) % ~Gray(?obj) &~Brown(?obj) &~Yellow(?obj) &~Red(?obj) &~Green(?obj) &~Purple(?obj) &~Cyan(?obj) ']].plot()
plt.show()

df[['forall ?obj: Small(?obj) % ~Large(?obj) ','forall ?obj: Large(?obj) % ~Small(?obj) ']].plot()
plt.show()

df[['forall ?obj: Cube(?obj) % ~Cube(?obj) &~Sphere(?obj) &~Cylinder(?obj) ']].plot()
plt.show()

df[['forall ?obj: Rubber(?obj) % ~Metal(?obj) ','forall ?obj: Metal(?obj) % ~Rubber(?obj) ']].plot()
plt.show()

df[['Right(object0,object1)','~Left(object0,object1)']].plot()
plt.show()

df[['forall ?obj, ?obj_2: Right(?obj, ?obj_2) % ~Left(?obj, ?obj_2)','forall ?obj, ?obj_2: Right(?obj, ?obj_2) % ~Right(?obj_2, ?obj)','forall ?obj: ~Right(?obj, ?obj)']].plot()
plt.show()