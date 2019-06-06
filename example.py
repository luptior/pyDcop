from pysbatch import *
import numpy as np


# directory
input="./simulation"
logdir="./log"

# generating data
colors_count=5
algo='dpop'
variables_count=10

cmd_list=["pydcop", "solve",
          "--algo", "dpop",
          "--run_metric", "empty",
          "--delay", 0,
          "empty"
]

cmd_list[cmd_list.index("--algo")+1]=algo

for variables_count in range(5, 85, 5):
    for delay in [0]:
        for instance in range(1, 3):
            print(f"running v{variables_count} delay{delay} insatnce{instance}")
            run_metric=f"{logdir}/metric_gc_{algo}_v{variables_count}_d{delay}_rep{instance}.csv"
            input_file=f"{input}/gc_v{variables_count}_c{colors_count}_rep{instance}.yaml"
            output_file=f"{logdir}/output_gc_{algo}_v{variables_count}_d{delay}_rep{instance}.yaml"

            print(run_metric)
            tmp_cmd = cmd_list
            tmp_cmd[tmp_cmd.index("--run_metric") + 1] = run_metric
            tmp_cmd[tmp_cmd.index("--delay") + 1] = str(delay)
            tmp_cmd[-1]=input_file

            output=run_cmd(tmp_cmd)

            with open(output_file, "w") as myf:
                myf.write(output)