import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        elif event.event_type in ['created']:
            print(f"File created or modified: {event.src_path}")
            execute_main_script()

def execute_streamlit():
    try:
        subprocess.run(["streamlit", "run", "app.py"])
        print("app.py executed successfully.")
    except Exception as e:
        print(f"Error executing app.py: {e}")

def execute_main_script():
    try:
        subprocess.run(["python", "main.py"])
        print("main.py executed successfully.")
    except Exception as e:
        print(f"Error executing main.py: {e}")

if __name__ == "__main__":
    folder_to_watch = "dataTemp/raw_temp"
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    try:
        print(f"Watching folder '{folder_to_watch}' for new files and modifications...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
