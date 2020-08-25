import matplotlib.pyplot as plt
import csv
import pandas as pd
import plotly.express as px
import math


pd.options.plotting.backend = 'plotly'

with open('axioms_values.csv', 'r') as f:
    df =  pd.read_csv(f)

df['sat_level'] = df.mean(numeric_only=True, axis=1)
#df['p_value'] = math.log10(10*df['sat_level'].data)

fig = px.line(df)
fig.update_layout(legend = {'x': -0.7, 'y':0, 'font':{'size':6}})

fig.show()


#fig.write_html("path/to/file.html")

#fig = px.line(df[['forall ?is_gray : Gray(?is_gray)','forall ?isnot_gray : ~Gray(?isnot_gray)']])
#fig.show()
# df[['forall ?is_gray : Gray(?is_gray)','forall ?isnot_gray : ~Gray(?isnot_gray)']].plot()
# plt.show()

#fig = px.line(df[['forall ?is_small : Small(?is_small)','forall ?isnot_small : ~Small(?isnot_small)','forall ?is_large : Large(?is_large)','forall ?isnot_large : ~Large(?isnot_large)']])
#fig.show()
# df[['forall ?is_small : Small(?is_small)','forall ?isnot_small : ~Small(?isnot_small)','forall ?is_large : Large(?is_large)','forall ?isnot_large : ~Large(?isnot_large)']].plot()
# plt.show()

#fig = px.line_matric(df[['forall ?is_cube : Cube(?is_cube)','forall ?isnot_cube : ~Cube(?isnot_cube)']])
#fig.show()
# df[['forall ?is_cube : Cube(?is_cube)','forall ?isnot_cube : ~Cube(?isnot_cube)']].plot()
# plt.show()


# df[['forall ?is_rubber : Rubber(?is_rubber)','forall ?isnot_rubber : ~Rubber(?isnot_rubber)','forall ?is_metal : Metal(?is_metal)','forall ?isnot_metal : ~Metal(?isnot_metal)']].plot()
# plt.show()

# df[['forall ?obj: Yellow(?obj) -> ~Gray(?obj) &~Blue(?obj) &~Brown(?obj) &~Red(?obj) &~Green(?obj) &~Purple(?obj) &~Cyan(?obj) ','forall ?obj: Small(?obj) -> ~Large(?obj) ','forall ?obj: Cube(?obj) -> ~Sphere(?obj) &~Cylinder(?obj) ','forall ?obj: Rubber(?obj) -> ~Metal(?obj) ']].plot()
# plt.show()

# df[['forall ?right_pair : Right(?right_pair)','forall ?left_pair : ~Right(?left_pair)', 'forall ?front_pair : Front(?front_pair)','forall ?behind_pair : ~Front(?behind_pair)']].plot()
# plt.show()