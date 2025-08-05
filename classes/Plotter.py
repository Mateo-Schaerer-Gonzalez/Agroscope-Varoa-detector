import os
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot



import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import cv2

class Plotter:
    def __init__(self, stage, output_folder, discobox_run):

        self.stage = stage
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # set the outputfolder
        if discobox_run:
            # get the output path
            self.output_path = os.path.abspath(os.path.join(base_dir, "..",  "..", output_folder))
        else:
            self.output_path = os.path.abspath(os.path.join(base_dir,"..",  output_folder))

        general_summary_path = os.path.abspath(os.path.join(self.output_path, os.pardir))


        self.pdf_path = os.path.join(self.output_path, "recording.pdf")
        self.csv_path = os.path.join(self.output_path, "summary.csv")
        self.frame_path = os.path.join(self.output_path, "frame_0.jpg")
        self.survival_path = os.path.join(self.output_path, "survival.png")
        self.time_survival_path = os.path.join(general_summary_path, "surivival.png")
        self.distribution_max_diff = os.path.join(general_summary_path, 'max_diff.png')
        self.distribution_max_local_diff = os.path.join(general_summary_path,'local_diff.png')
        self.distribution_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variabilites.csv')
   
    
    def make_survival_graph(self):
        summary_data = self.stage.data
        all_maxdiff = self.stage.max_diff

       
        zone_labels = summary_data['Zone ID']
    
        survival_rates = summary_data['Survival %']

        # Create a figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: Histogram of max pixel difference
        axes[0].hist(all_maxdiff, bins=20, color="steelblue", edgecolor="black")
        axes[0].set_title("Distribution of Mite Max Pixel Difference")
        axes[0].set_xlabel("Max Pixel Difference")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True)

        # Subplot 2: Bar chart of survival rates
        axes[1].bar(zone_labels, survival_rates, color="mediumseagreen", edgecolor="black")
        axes[1].set_title("Survival Rate by Zone")
        axes[1].set_xlabel("Zone ID")
        axes[1].set_ylabel("Survival %")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.survival_path)
        plt.close()

    
    def create_recording_pdf(self, recording_count):
        
        with PdfPages(self.pdf_path) as pdf:
            # A4 size in inches: 8.27 x 11.69
            fig = plt.figure(figsize=(8.27, 11.69))

           
            n_rows = 3
            gs = fig.add_gridspec(n_rows, 1, height_ratios=[1, 3, 1.5])

            # --- Section 1: Summary Table ---
            
            df = self.stage.data
            df = df[df['recording'] == recording_count]
            print(df.head())
            ax1 = fig.add_subplot(gs[0])
            ax1.axis('off')
            table = ax1.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.0)
            ax1.set_title("Zone Survival Summary", fontsize=12, pad=10)

            # --- Section 2: Frame Image ---
            if os.path.exists(self.frame_path):
                img = plt.imread(self.frame_path)
                ax2 = fig.add_subplot(gs[1])
                ax2.imshow(img)
                ax2.axis('off')
                ax2.set_title("Detection output", fontsize=12, pad=10)

            # --- Section 3: Survival Plot ---
            if os.path.exists(self.survival_path):
                img2 = plt.imread(self.survival_path)
                ax3 = fig.add_subplot(gs[2])
                ax3.imshow(img2)
                ax3.axis('off')
                ax3.set_title("Survival Rate by Zone", fontsize=12, pad=10)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF successfully saved to: {self.pdf_path}")


    def make_survival_time_graph(self):
        df = self.stage.data
        plt.figure(figsize=self.stage.img_size)

        plt.figure(figsize=(10, 6))

        for zone in df['Zone ID'].unique():
            zone_data = df[df['Zone ID'] == zone]
            plt.plot(
                zone_data['recording'],
                zone_data['Survival %'],
                marker='o',
                label=f'{zone}'
            )

        plt.xlabel('Recording Count')
        plt.ylabel('Survival %')
        plt.title('Survival Percentage Over Recordings Per Zone')
        plt.legend(title='Zone ID')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.time_survival_path)
        plt.close()



    def distribution_graph(self):
        df = pd.read_csv(self.distribution_data_path)

        # Plot histogram for each group (Alive True/False)
        plt.figure(figsize=self.stage.img_size)

        for alive_status, group in df.groupby('Ground_truth'):
            plt.hist(group['max_diff'], bins=30, alpha=0.6, label=alive_status)

        
        plt.xlabel('max difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of local difference')
        plt.legend()
        plt.savefig(self.distribution_max_diff)
        plt.close()


        for alive_status, group in df.groupby('Ground_truth'):
            plt.hist(group['local_diff'], bins=30, alpha=0.6, label=alive_status)


        plt.xlabel('local difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of local difference')
        plt.legend()
        plt.savefig(self.distribution_max_local_diff)
        plt.close()

    def save_frame0_detection(self, image, thickness=2):
        # Draw zones on the image
        for zone in self.stage.zones:
            zone.draw(image, thickness=thickness)

        # Save image with a unique name
        filename = os.path.join(self.output_path, "frame_0.jpg")

        cv2.imwrite(filename, image)
        print(f"Image saved to: {filename}")


