# test_prefect_tags.py
import prefect
print(f"Prefect version in test script: {prefect.__version__}")

from prefect import flow

@flow(name="test_flow", tags=["test:tag"])
def simple_test_flow():
    print("Test flow ran!")

if __name__ == "__main__":
    simple_test_flow()