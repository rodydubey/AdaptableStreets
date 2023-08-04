
import os

densities = [1.75,
            4.87,
            9.27,
            16.69,
            35.64,
            ]


# for density in densities:
#     os.system(f"python testing_main.py --density {density} --num_seeds {10}")

# density = 4.87
# modeltypes = ['heuristic', 'static']

# for modeltype in modeltypes:
#     # for traffic in traffic_conditions:
#     os.system(f"python testing_main.py --density {density} --num_seeds {10} --modeltype {modeltype}")


# density = 4.87
# modeltype = 'model'
# edges = ['E0','-E1','-E2','-E3']
# print(edges)
# for joint in [True, False]:
#     # for traffic in traffic_conditions:

#     call = f"python testing_main.py --density {density} --num_seeds {1} --modeltype {modeltype} --edges {','.join(edges)}"
#     if joint:
#         call += ' --joint_agents'
#     print(call)
#     os.system(call + "> /dev/null 2>&1")
