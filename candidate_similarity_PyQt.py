import os
import sys
from pinecone import Pinecone
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QSizePolicy

from Pinecone_processing import read_local_file, resume_to_vector, find_similar_resumes, chat_similar_resumes

class ResumeUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.uploadButton = QPushButton('Upload Resume', self)
        self.uploadButton.clicked.connect(self.upload_resume)
        self.layout.addWidget(self.uploadButton)

        self.filePreview = QTextEdit(self)
        self.filePreview.setReadOnly(True)
        self.layout.addWidget(self.filePreview)

        self.resultLabel = QLabel('Similar candidates will be shown here.', self)
        self.layout.addWidget(self.resultLabel)

        self.layout.setStretch(1, 2)
        self.layout.setStretch(2, 1)

        self.setLayout(self.layout)
        self.setWindowTitle('Resume Similarity Checker')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.showMaximized()
        self.show()

    def upload_resume(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload Resume", "",
                                                  "PDF Files (*.pdf);;DOCX Files (*.docx);;All Files (*)",
                                                  options=options)
        if fileName:
            self.preview_file(fileName)
            self.process_resume(fileName)

    def preview_file(self, file_path):
        content = read_local_file(file_path)
        self.filePreview.setText(content)

    def process_resume(self, file_path):
        resume_vector = resume_to_vector(index, file_path)

        # ==== METHOD 1 ==== #
        # similar_resumes = find_similar_resumes(index, resume_vector)
        #
        # result_text = "Similar candidates:\n"
        # for match in similar_resumes:
        #     result_text += f"ID: {match.metadata['id']}\n"

        # ==== METHOD 2 ==== #
        result_text = chat_similar_resumes(index, resume_vector)
        self.resultLabel.setText(result_text)


if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = ""
    os.environ['PINECONE_API_KEY'] = ""
    pc = Pinecone()
    index = pc.Index('queryfiles')


    app = QApplication(sys.argv)
    ex = ResumeUploader()
    sys.exit(app.exec_())
