import lightning as L
import subprocess


class MyWork(L.LightningWork):
    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute("cpu"), parallel=True)

    def run(self):
        subprocess.run(
            f"""
            python multitask/suzuki_optimization.py stbo baumgartner_suzuki benchmark_baumgartner_suzuki:latest data/baumgartner_suzuki/results_stbo_mixed_domain  --brute-force-categorical --max-experiments 20 --batch-size 1 --repeats 1 --wandb-artifact-name stbo_baumgartner_suzuki
            """,
            shell=True,
        )


class MyFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.my_work = MyWork()

    def run(self):
        self.my_work.run()


if __name__ == "__main__":
    app = L.LightningApp(MyFlow())


# class Work(L.LightningWork):
#     def __init__(self, *args, **kwargs):
#         super().__init__(parallel=True)

#     def run(self, *args, **kwargs):
#         print("Hello world from work")


# class Flow(L.LightningFlow):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.work = Work()

#     def run(self):
#         self.work.run()


# if __name__ == "__main__":
#     app = L.LightningApp(Flow())
