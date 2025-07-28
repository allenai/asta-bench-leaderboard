from content import LIT_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Literature Understanding"

def build_page():
    build_category_page(CATEGORY_NAME, LIT_DESCRIPTION)
