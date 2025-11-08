import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog # <--- IMPORT filedialog
import tkinterdnd2 as TkinterDnD 
from sentiment_model import SentimentModel 
import os 

class SentimentApp(TkinterDnD.Tk): 
    """
    This class creates the Tkinter GUI window and
    connects the buttons to the SentimentModel logic.
    """
    def __init__(self, model: SentimentModel):
        super().__init__()
        
        self.model = model
        
        # --- NEW: Add a variable to store the selected file path ---
        self.batch_file_path = tk.StringVar()
        
        # --- 1. Customize Title and Size ---
        self.title("My Custom Sentiment Analyzer")
        self.geometry("700x550") # Made the window a bit bigger
        
        # --- 2. Customize Theme ---
        self.style = ttk.Style(self)
        self.style.theme_use('alt') 

        # --- 3. Customize Colors and Fonts (Advanced) ---
        self.style.configure(
            'TFrame', 
            background='#f0f0f0'  # Light gray background for tabs
        )
        self.style.configure(
            'TLabel', 
            background='#f0f0f0',  # Match tab background
            font=('Arial', 11)   # Default font for all labels
        )
        self.style.configure(
            'TButton', 
            font=('Arial', 10, 'bold'),
            foreground='white',
            background='#00529b'  # A nice blue color
        )
        self.style.map(
            'TButton',
            background=[('active', '#003p7a')] # Darker blue on hover
        )
        self.style.configure(
            'TNotebook.Tab',
            font=('Arial', 10, 'bold'),
            padding=[10, 5]  # Add some padding to the tab headers
        )


        # --- Create the main layout (Tabs) ---
        self.notebook = ttk.Notebook(self)
        
        self.frame_analyze = ttk.Frame(self.notebook, padding=15)
        self.frame_manage = ttk.Frame(self.notebook, padding=15)
        self.frame_add = ttk.Frame(self.notebook, padding=15)
        self.frame_batch = ttk.Frame(self.notebook, padding=15)
        
        self.notebook.add(self.frame_analyze, text='Analyze Website')
        self.notebook.add(self.frame_manage, text='Add New Site')
        self.notebook.add(self.frame_add, text='Add Single Comment')
        self.notebook.add(self.frame_batch, text='Batch Upload')
        
        self.notebook.pack(expand=True, fill='both')

        # --- Populate the "Analyze Website" Tab ---
        self.create_analyze_tab()
        
        # --- Populate the "Add New Site" Tab ---
        self.create_manage_tab() 
        
        # --- Populate the "Add Comment" Tab ---
        self.create_add_tab()
        
        # --- Populate the "Batch Upload" Tab ---
        self.create_batch_tab()
        
        # --- Load initial data ---
        self.refresh_site_dropdowns()

    def create_analyze_tab(self):
        # Configure grid for padding
        self.frame_analyze.columnconfigure(0, weight=1)
        self.frame_analyze.columnconfigure(1, weight=0)
        self.frame_analyze.columnconfigure(2, weight=1)

        # Label
        lbl_select = ttk.Label(self.frame_analyze, text="Select Website ID to Analyze:")
        lbl_select.grid(row=0, column=1, pady=(10, 5), sticky='sw')
        
        # Dropdown (Combobox)
        self.analyze_site_var = tk.StringVar()
        self.combo_analyze = ttk.Combobox(
            self.frame_analyze, 
            textvariable=self.analyze_site_var,
            state="readonly",
            width=60
        )
        self.combo_analyze.grid(row=1, column=1, pady=5, sticky='ew')
        
        # Button
        btn_analyze = ttk.Button(
            self.frame_analyze, 
            text="Calculate WSI", 
            command=self.on_analyze_click
        )
        btn_analyze.grid(row=2, column=1, pady=10)
        
        # --- NEW: Status Label ---
        self.analyze_status_var = tk.StringVar()
        lbl_analyze_status = ttk.Label(
            self.frame_analyze,
            textvariable=self.analyze_status_var,
            font=("Arial", 10, "italic"),
            background='#f0f0f0'
        )
        lbl_analyze_status.grid(row=3, column=1, pady=(0, 10))
        # --- END NEW ---
        
        # Results Text Area
        lbl_results = ttk.Label(self.frame_analyze, text="Results:")
        lbl_results.grid(row=4, column=1, pady=(10, 5), sticky='sw')
        
        self.text_results = scrolledtext.ScrolledText(
            self.frame_analyze, 
            wrap=tk.WORD, 
            height=15, 
            width=80, 
            font=("Consolas", 10)
        )
        self.text_results.grid(row=5, column=1, sticky='nsew')
        
        # Configure the row containing the text area to expand vertically
        self.frame_analyze.rowconfigure(5, weight=1)

    # --- FIX: Renamed function to create_manage_tab ---
    def create_manage_tab(self):
        """Creates the content for the 'Add New Site' tab."""
        # Configure grid for padding
        self.frame_manage.columnconfigure(0, weight=1)
        self.frame_manage.columnconfigure(1, weight=0)
        self.frame_manage.columnconfigure(2, weight=1)

        # Label
        lbl_new_site = ttk.Label(self.frame_manage, text="Enter New Website ID:")
        lbl_new_site.grid(row=0, column=1, padx=5, pady=(10, 5), sticky='sw')

        # Entry
        self.new_site_var = tk.StringVar()
        entry_new_site = ttk.Entry(self.frame_manage, textvariable=self.new_site_var, width=60, font=("Arial", 11))
        entry_new_site.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Button
        btn_add_site = ttk.Button(
            self.frame_manage, 
            text="Add New Website", 
            command=self.on_add_website_click
        )
        # --- FIX: Changed btn_analyze to btn_add_site ---
        btn_add_site.grid(row=2, column=1, pady=20)
        
        # Status Label
        self.manage_status_var = tk.StringVar()
        lbl_status = ttk.Label(
            self.frame_manage, 
            textvariable=self.manage_status_var, 
            font=("Arial", 10, "italic"),
            background='#f0f0f0'
        )
        lbl_status.grid(row=3, column=1, pady=10)

    # --- FIX: Re-added the correct create_add_tab ---
    def create_add_tab(self):
        """Creates the content for the 'Add Single Comment' tab."""
        # Configure grid for resizing
        self.frame_add.columnconfigure(1, weight=1)
        self.frame_add.rowconfigure(1, weight=1) # Make text box expandable

        # Site ID
        lbl_add_site = ttk.Label(self.frame_add, text="Website ID:")
        lbl_add_site.grid(row=0, column=0, padx=5, pady=10, sticky='w')
        
        self.add_site_var = tk.StringVar()
        self.combo_add = ttk.Combobox(
            self.frame_add, 
            textvariable=self.add_site_var,
            width=40
        )
        self.combo_add.grid(row=0, column=1, padx=5, pady=10, sticky='ew')
        
        # Comment
        lbl_add_comment = ttk.Label(self.frame_add, text="Comment:")
        lbl_add_comment.grid(row=1, column=0, padx=5, pady=10, sticky='nw')
        
        self.entry_comment = tk.Text(
            self.frame_add, 
            height=5,
            width=42, 
            font=("Arial", 11),
            wrap=tk.WORD
        ) 
        self.entry_comment.grid(row=1, column=1, padx=5, pady=10, sticky='nsew')
        
        # Button
        btn_add = ttk.Button(
            self.frame_add, 
            text="Add Comment to CSV", 
            command=self.on_add_comment_click
        )
        btn_add.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Status Label
        self.add_status_var = tk.StringVar()
        lbl_status = ttk.Label(
            self.frame_add, 
            textvariable=self.add_status_var, 
            font=("Arial", 10, "italic"),
            background='#f0f0f0'
        )
        lbl_status.grid(row=3, column=0, columnspan=2, pady=10)

    # --- FIX: Renamed function to create_batch_tab ---
    def create_batch_tab(self):
        """Creates the content for the 'Batch Upload' tab."""
        # Configure grid for centering and resizing
        self.frame_batch.columnconfigure(0, weight=1)
        self.frame_batch.columnconfigure(1, weight=1)
        self.frame_batch.columnconfigure(2, weight=1)
        self.frame_batch.rowconfigure(4, weight=1) # Make drop zone expandable

        # 1. Site ID Label
        lbl_batch_site = ttk.Label(self.frame_batch, text="1. Select Website ID to add comments to:")
        lbl_batch_site.grid(row=0, column=1, padx=5, pady=(10, 5), sticky='sw')
        
        # 2. Site ID Dropdown
        self.batch_site_var = tk.StringVar()
        self.combo_batch = ttk.Combobox(
            self.frame_batch, 
            textvariable=self.batch_site_var,
            state="readonly",
            width=60
        )
        self.combo_batch.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # 3. File Selection Label
        lbl_batch_drop = ttk.Label(self.frame_batch, text="2. Select a .txt file:")
        lbl_batch_drop.grid(row=2, column=1, padx=5, pady=(20, 5), sticky='sw')

        # 4. Frame for Browse Button and Entry Box
        browse_frame = ttk.Frame(self.frame_batch)
        browse_frame.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        browse_frame.columnconfigure(0, weight=1) # Make entry expand
        
        entry_file_path = ttk.Entry(
            browse_frame, 
            textvariable=self.batch_file_path, 
            state="readonly", 
            width=50
        )
        entry_file_path.grid(row=0, column=0, sticky='ew')
        
        btn_browse = ttk.Button(
            browse_frame, 
            text="Browse...", 
            command=self.on_browse_file
        )
        btn_browse.grid(row=0, column=1, padx=(5, 0))

        # 5. Drop Zone
        self.drop_target = tk.Label(
            self.frame_batch, 
            text="OR Drag and Drop file here",
            relief="solid", 
            borderwidth=2, 
            background="#ffffff",
            foreground="#aaaaaa",
            font=("Arial", 10, "italic"),
            pady=10
        )
        self.drop_target.grid(row=4, column=1, padx=5, pady=5, sticky='nsew')
        
        # 6. Register the drop zone
        self.drop_target.drop_target_register(TkinterDnD.DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self.on_file_drop)
        
        # 7. Submit Button
        btn_submit_batch = ttk.Button(
            self.frame_batch,
            text="Submit Batch File",
            command=self.on_submit_batch
        )
        btn_submit_batch.grid(row=5, column=1, padx=5, pady=15)
        
        # 8. Status Label (Now row 6)
        self.batch_status_var = tk.StringVar()
        lbl_batch_status = ttk.Label(
            self.frame_batch, 
            textvariable=self.batch_status_var, 
            font=("Arial", 10, "italic"),
            background='#f0f0f0'
        )
        lbl_batch_status.grid(row=6, column=1, pady=10)

    def refresh_site_dropdowns(self):
        """Fetches the current site list from the model and updates ALL dropdowns."""
        sites = self.model.get_available_sites()
        
        self.combo_analyze['values'] = sites
        self.combo_add['values'] = sites
        self.combo_batch['values'] = sites # Update batch dropdown
        
        # Set a default selection if possible
        if sites:
            # Set defaults only if they aren't already set to something valid
            if not self.analyze_site_var.get() in sites:
                self.analyze_site_var.set(sites[0])
            if not self.add_site_var.get() in sites:
                self.add_site_var.set(sites[0])
            if not self.batch_site_var.get() in sites: # Set batch default
                self.batch_site_var.set(sites[0])

    def on_analyze_click(self):
        """Handles the 'Calculate WSI' button click."""
        site_id = self.analyze_site_var.get()
        if not site_id:
            messagebox.showwarning("No Site", "Please select a website to analyze.")
            return

        # Clear status
        self.analyze_status_var.set("") 
        
        # 1. Clear previous results
        self.text_results.delete('1.0', tk.END)
        self.text_results.insert('1.0', f"Analyzing '{site_id}'...")
        self.update_idletasks()
        
        results = self.model.get_website_analysis(site_id)
        
        self.text_results.delete('1.0', tk.END)

        if "error" in results:
            self.text_results.insert('1.0', f"Error: {results['error']}")
            # Show error status
            self.analyze_status_var.set(f"Error: {results['error']}")
            return

        # Show file save status
        if results.get("summary_file_path"):
            self.analyze_status_var.set(f"Summary saved to {results['summary_file_path']}")
        else:
            self.analyze_status_var.set("Error: Could not save summary file.")

        wsi = results['wsi']
        total = results['total_comments']
        counts = results['counts']
        count_pos = counts.get('Positive', 0)
        count_neu = counts.get('Neutral', 0)
        count_neg = counts.get('Negative', 0)

        output = f"--- Analysis Results for '{site_id}' ---\n\n"
        output += f"Website Satisfaction Index (WSI): {wsi:.2f}\n"
        output += "----------------------------------------\n"
        output += f"Total Comments Analyzed: {total}\n"
        
        if total > 0:
            output += f"Positive Comments: {count_pos} ({count_pos/total:.1%})\n"
            output += f"Neutral Comments:  {count_neu} ({count_neu/total:.1%})\n"
            output += f"Negative Comments: {count_neg} ({count_neg/total:.1%})\n\n"
        
        output += "--- Labeled Data ---\n"
        output += results['labeled_dataframe'][['User_Comment', 'Sentiment']].to_string(index=False)
        
        self.text_results.insert('1.0', output)

    def on_add_website_click(self):
        """Handles the 'Add New Website' button click."""
        new_site_id = self.new_site_var.get().strip()
        
        if not new_site_id:
            messagebox.showwarning("Missing Info", "Please enter a Website ID.")
            return

        status_message = self.model.add_website(new_site_id)
        self.manage_status_var.set(status_message)
        
        if "Successfully" in status_message:
            self.new_site_var.set("")
            self.refresh_site_dropdowns()
            self.add_site_var.set(new_site_id)
            self.analyze_site_var.set(new_site_id)
            self.batch_site_var.set(new_site_id) 

    def on_add_comment_click(self):
        """Handles the 'Add Comment' button click."""
        site_id = self.add_site_var.get().strip()
        comment = self.entry_comment.get("1.0", "end-1c").strip()
        
        if not site_id or not comment:
            messagebox.showwarning("Missing Info", "Please provide both a Website ID and a comment.")
            return
            
        status_message = self.model.add_comment_to_csv(site_id, comment)
        self.add_status_var.set(status_message)
        
        if "Successfully" in status_message:
            self.entry_comment.delete("1.0", tk.END)
            # Only refresh if the site was new
            if site_id not in self.combo_add['values']:
                self.refresh_site_dropdowns()
            self.add_site_var.set(site_id) # Re-set the site
            
    def on_browse_file(self):
        """Opens a file dialog to select a .txt file."""
        file_path = filedialog.askopenfilename(
            title="Select a .txt file",
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            self.batch_file_path.set(file_path)

    def on_submit_batch(self):
        """Handles the 'Submit Batch File' button click."""
        file_path = self.batch_file_path.get()
        self.process_batch_file(file_path) # Call helper

    def on_file_drop(self, event):
        """
File `sentiment_gui.py` updated.
        Handles the event when a file is dropped onto the drop zone.
        """
        file_path = event.data.strip('{}')
        self.batch_file_path.set(file_path) # Set the file path in the entry
        self.process_batch_file(file_path) # Call helper

    def process_batch_file(self, file_path: str):
        """
        Contains the core logic for processing a batch file,
        called by both on_file_drop and on_submit_batch.
        """
        site_id = self.batch_site_var.get().strip()

        if not site_id:
            messagebox.showwarning("No Site", "Please select a Website ID from the dropdown.")
            self.batch_status_var.set("Error: No Website ID selected.")
            return
            
        if not file_path:
            messagebox.showwarning("No File", "Please select a file first.")
            self.batch_status_var.set("Error: No file selected.")
            return

        if not file_path.endswith('.txt'):
            messagebox.showwarning("Invalid File", "Please select a .txt file only.")
            self.batch_status_var.set("Error: File must be a .txt file.")
            return

        try:
            self.batch_status_var.set(f"Processing '{os.path.basename(file_path)}'...")
            self.update_idletasks()
            
            status_message = self.model.batch_add_comments_from_file(site_id, file_path)
            
            self.batch_status_var.set(status_message)
            self.refresh_site_dropdowns()
            
            # Clear file path from entry box on success
            if "Successfully" in status_message:
                self.batch_file_path.set("")
            
        except Exception as e:
            status_message = f"Error processing file: {e}"
            self.batch_status_var.set(status_message)
            messagebox.showerror("Error", status_message)


def main():
    try:
        model = SentimentModel()
        app = SentimentApp(model)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Critical Error", f"Failed to start application:\n{e}")
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()