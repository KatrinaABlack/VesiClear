import time

class TimeTracker:
    def __init__(self):
        """
        Initialize a time tracker object, to measure task runtimes.
        """
        self.start_times = {}
        self.runtimes = {}
    
    def start_task(self, task: str) -> None:
        """
        Start a new task with the given name.
        """
        if task in self.start_times:
            print(f"Task {task} already running! Start time reset")
        self.start_times[task] = time.time()

    def stop_task(self, task: str) -> None:
        """
        Stop a running task and add its runtime to the tracked duration for that task.
        """
        if task not in self.start_times:
            raise Exception(f"Task {task} was not started!")
        if task not in self.runtimes:
            self.runtimes[task] = 0
        self.runtimes[task] += time.time() - self.start_times[task]
        del self.start_times[task]

    def get_runtimes(self) -> dict[str, float]:
        """
        Get the runtimes of all tasks.
        """
        return self.runtimes
    
    def add_runtimes(self, add_runtimes: dict[str, float]) -> None:
        """
        Add a dictionary of task runtimes to the tracked runtimes.
        """
        for task, runtime in add_runtimes.items():
            if task not in self.runtimes:
                self.runtimes[task] = 0
            self.runtimes[task] += runtime

    def stop_running_tasks(self) -> None:
        """
        Stop all running tasks.
        """
        # Get frozen list of tasks, to modify dictionary while iterating
        running_tasks = list(self.start_times.keys())
        for task in running_tasks:
            self.stop_task(task)

    def __str__(self) -> str:
        """
        Return a string of all runtimes recorded in the time tracker.
        """
        out_str = ""
        for task, runtime in self.runtimes.items():
            out_str += f"Time (s) in {task}: {runtime}\n"
        return out_str
