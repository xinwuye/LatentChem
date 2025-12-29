import re

INPUT_FILE = "./extracted/stage3/final_outputs_with_indices.txt"
EXTRA_CHECK_FILE = "./extracted/stage3/extra_check.txt"

def filter_txt_entries():
    print(f"Opening {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Split by double newline to get each entry (paragraph + index)
        content = f.read().strip()
        entries = content.split('\n\n')

    main_list = []
    extra_check_list = []

    # Patterns to look for at the start of the string
    # Matches "In conclusion", "In summary", etc. 
    valid_starts = ("In conclusion", "In summary", "To summarize", "Thus", "In sum", "Overall", "Therefore", 
                    "Answer:", "Final answer:", "**Answer**:", "### Conclusion", "**Answer:**", "**Conclusion**:", 
                    "As a result", "To conclude", "Collectively")

    for entry in entries:
        clean_entry = entry.strip()
        if not clean_entry:
            continue
            
        # 1. Check your phrase list first
        starts_with_phrase = clean_entry.lower().startswith(tuple(s.lower() for s in valid_starts))
        
        # 2. Check for the **bold** pattern at the very start
        # ^\*\* : Starts with **
        # .+?       : Matches any characters (short as possible)
        # \*\* : Ends with **
        bold_pattern = bool(re.match(r'^\*\*.{1,15}?\*\*', clean_entry))

        if starts_with_phrase or bold_pattern:
            main_list.append(clean_entry)
        else:
            extra_check_list.append(clean_entry)
        
    # Save the "failed" entries
    with open(EXTRA_CHECK_FILE, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(extra_check_list))
    
    print(f"Processing Complete:")
    print(f" - Standardized: {len(main_list)} entries")
    print(f" - Moved to extra_check: {len(extra_check_list)} entries")

if __name__ == "__main__":
    filter_txt_entries()