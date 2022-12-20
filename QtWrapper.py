import os.path
import sys
from os.path import basename
import time
import BookAIShell
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class Picture(QDialog):
    def __init__(self, path, size, parent=None,):
        super(Picture, self).__init__(parent)
        layout = QVBoxLayout(self)

        img_label = QLabel("")
        img = QPixmap(path)
        img = img.scaledToWidth(size)
        img_label.setPixmap(img)
        #img = QPixmap('C:/Users/Loki/Desktop/output-2/outputPlotWithInfo.jpg')
        #img_label.setPixmap('C:/Users/Loki/Desktop/output-2/outputPlotWithInfo.jpg')
        img_label.show()

        layout.addWidget(img_label)

class Notification(QDialog):
    def __init__(self, parent=None, ):
        super(Notification, self).__init__(parent)
        layout = QVBoxLayout(self)

        notif = QLabel("Recognition finished!")
        notif.show()

        layout.addWidget(notif)

class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.window_width = 500
        self.max_text_len = 80
        self.setWindowTitle("Bookshelf AI")

        #self.input_string = "./segmentation/images/im0.jpg"
        #self.output_string = "pipeline_test_output/shelltesting" + time.strftime("%m%d%H%M%S")

        self.input_string = "..."
        self.output_string = "..."

        layout = QVBoxLayout(self)

        self.form_group_box = self.create_in_out_grid()
        self.run_box = self.create_run_box()

        layout.addWidget(self.form_group_box)
        layout.addWidget(self.run_box)

    def create_in_out_grid(self):
        s = "this is a very long test string, too long, in fact, to fit into the label box I create (for test purposes). " \
            "I'm trying to see what happens when the string gets way too long!!!"
        formGroupBox = QGroupBox("Input and Output")
        layout = QFormLayout()
        browse_file = QPushButton("Browse file...")
        browse_file.clicked.connect(self.find_file)
        browse_folder = QPushButton("Browse folder...")
        browse_folder.clicked.connect(self.find_directory)
        layout.addRow(QLabel("Path to input image: "), browse_file)
        self.input_box = QLabel(self.input_string)
        self.input_box.setAlignment(Qt.AlignRight)
        self.input_box.setStyleSheet("QLabel { background-color : white; color : black; }")
        layout.addRow(self.input_box)
        out = QLabel("Path to output folder: ")
        out.setFixedWidth(self.window_width-100)
        layout.addRow(out, browse_folder)
        self.output_box = QLabel(self.output_string)
        self.output_box.setAlignment(Qt.AlignRight)
        self.output_box.setStyleSheet("QLabel { background-color : white; color : black; }")
        layout.addRow(self.output_box)
        formGroupBox.setLayout(layout)
        return formGroupBox

    def create_run_box(self):
        formGroupBox = QGroupBox("Run the program!")
        layout = QVBoxLayout()
        self.run_program = QPushButton("Recognize books...")
        self.run_program.clicked.connect(self.run)
        if not (os.path.isfile(self.input_string) and os.path.isdir(self.output_string)):
            self.run_program.setEnabled(False)
        layout.addWidget(self.run_program)
        formGroupBox.setLayout(layout)
        return formGroupBox

    def shorten_text(self, text):
        if len(text) > self.max_text_len:
            text = "..." + text[(len(text)-self.max_text_len+3):]
        return text

    def find_file(self):
        s = QFileDialog.getOpenFileName(self, "Select input image", "", "Images (*.jpg)")
        self.input_string = str(s[0])
        self.input_box.setText(self.shorten_text(self.input_string))

        if os.path.isfile(self.input_string) and os.path.isdir(self.output_string):
            self.run_program.setEnabled(True)

    def find_directory(self):
        s = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        self.output_string = str(s)
        self.output_box.setText(self.shorten_text(self.output_string))

        if os.path.isfile(self.input_string) and os.path.isdir(self.output_string):
            self.run_program.setEnabled(True)

    def run(self):
        image = basename(self.input_string)
        print(self.input_string)
        print(self.input_string.split(image)[0])
        print(image)
        args = ["bookai", self.input_string.split(image)[0], image, self.output_string]

        before = time.time()

        shell = BookAIShell.BookAIShell(args)
        #print("THIS IS A REPLACEMENT FOR THE ACTUAL PIPELINE (FOR TESTING)")

        #new_form = Picture(self.output_string+"/outputPlotWithInfo.jpg", 1400, self)
        #new_form.show()

        elapsed_time = "TOTAL ELAPSED TIME: {}".format(time.time() - before)
        line = "-" * len(elapsed_time) + "-" * 22
        spaces = " " * 10
        print(line)
        print("|" + spaces + elapsed_time + spaces + "|")
        print(line)

        popup = Notification(self)
        popup.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)

    form = Form()

    form.resize(form.window_width, 200)
    form.show()

    sys.exit(app.exec())
