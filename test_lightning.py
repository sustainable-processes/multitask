import lightning as L
import subprocess

class MyWork(L.LightningWork):
    def __init__(self, batch_size: int, parallel: bool ):
        super().__init__(parallel=parallel)
        self.batch_size = batch_size

    def run(self):
        subprocess.run(
            f"""
            python multitask/suzuki_optimization.py stbo baumgartner_suzuki benchmark_baumgartner_suzuki:latest data/baumgartner_suzuki/results_stbo_mixed_domain  --brute-force-categorical --max-experiments 20 --batch-size {self.batch_size} --repeats 1 --wandb-artifact-name stbo_baumgartner_suzuki
            """,
            shell=True,
            # executable
        )

class MyFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.my_work = MyWork()

    def run(self):
        self.my_work.run()




# if __name__=="__main__":
#     # app = L.LightningApp(MyFlow())

# import lightning as L
# # basic hello World
# class Root(L.LightningFlow):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def run(self,*args, **kwargs):
#         print("Hello world from flow's infinite event loop")
if __name__ == "__main__":
    app = L.LightningApp(MyFlow())