import os
import streamlit as st
from helpers import *

def main():
    st.set_page_config(page_title="Researcher...", 
                       page_icon=":parrot:", 
                       layout="wide")
    
    st.header("Generate a Newsletter :parrot:")
    query = st.text_input("Enter a topic...")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    num_articles = st.number_input("Number of Articles", min_value=1, max_value=10, value=3, step=1)
    custom_sections = st.text_input("Custom Newsletter Sections (comma-separated)")
    generate_newsletter_btn = st.button("Generate Newsletter")
    
    if generate_newsletter_btn:
        print(query)
        st.write(query)
        with st.spinner(f"Generating newsletter for {query}"):
            search_results = search_serp(query=query, start_date=start_date, end_date=end_date)
            urls = pick_best_articles_urls(response_json=search_results, query=query, start_date=start_date, end_date=end_date, num_articles=num_articles)
            data = extract_content_from_urls(urls)
            summaries = summarizer(data, query)
            newsletter_thread = generate_newsletter(summaries, query, custom_sections)
            
            with st.expander("Search Results"):
                st.info(search_results)
            with st.expander("Best URLs"):
                st.info(urls)
            with st.expander("Data"):
                data_raw = " ".join(d.page_content for d in data.similarity_search(query,k=4))
                st.info(data_raw)
            with st.expander("Summaries"):
                st.info(summaries)
            with st.expander("Newsletter:"):
                st.info(newsletter_thread)
            
            if st.button("Send Newsletter"):
                # Placeholder for email integration
                st.success("Newsletter sent successfully!")
        st.success("Done!")

if __name__ == '__main__':
    main()