import sys
 
def wait_for_enter():
    input("Press Enter to continue: ")

class DVCInit(object):
    def run(self, context):
        print("Open a second terminal, in the same folder")
        print("Please note that DVC is already added in the gitpod. If running on your own computer make sure to install requirements or validate dvc.org.")
        print("Let's initialize DVC")
        print(" $ dvc init")
        wait_for_enter()
        print(" $ git add .")
        wait_for_enter()
        print(" $ git commit 'dvc initalized'")
        wait_for_enter()

class DVCAddPipelineStep(object):
    def run(self, context):
        print("Let's add our first pipeline step")
        print("-n <name>")
        print("-d <dependenc/y/ies>")
        print("-o <output>")
        print("cmd")
        print(" $ dvc run -n download_data -d download_data.sh -o lol_dataset download_data.sh")
        wait_for_enter()
        print(" $ git add .")
        wait_for_enter()
        print(" $ git commit 'data added with first pipeline step'")
        wait_for_enter()
        print(" $ dvc repro")
        wait_for_enter()

if __name__ == "__main__":
    context = {"username": sys.argv[1]}
    procedure = [
        DVCInit(),
        DVCAddPipelineStep()
    ]
    for step in procedure:
        step.run(context)
    print("Done.")