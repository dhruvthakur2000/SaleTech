from src.saletech.utils.logger import setup_logging, get_logger
import time

setup_logging()
logger = get_logger("saletech.demo")

for i in range(3):
    logger.info("demo_log", iteration=i)
    time.sleep(0.5)
logger.error("demo_error", error="This is a test error log!")
print("Demo logging complete. Check logs directory.")
