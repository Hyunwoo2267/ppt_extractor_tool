import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
from pathlib import Path
import subprocess
import sys


# 기존 PPTExtractor 클래스를 임포트한다고 가정
# from ppt_extractor import PPTExtractor

class PPTExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPT 슬라이드 추출 툴")
        self.root.geometry("600x700")
        self.root.resizable(True, True)

        # 변수 초기화
        self.video_path = tk.StringVar()
        self.youtube_url = tk.StringVar()
        self.output_dir = tk.StringVar(value="extracted_slides")
        self.similarity_threshold = tk.DoubleVar(value=0.95)
        self.frame_interval = tk.IntVar(value=3)
        self.is_processing = False

        # 로그 큐 (스레드 간 통신용)
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 제목
        title_label = ttk.Label(main_frame, text="PPT 슬라이드 추출 툴",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 입력 방법 선택
        input_frame = ttk.LabelFrame(main_frame, text="입력 방법 선택", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # 탭 생성
        notebook = ttk.Notebook(input_frame)
        notebook.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # 로컬 파일 탭
        file_frame = ttk.Frame(notebook, padding="10")
        notebook.add(file_frame, text="로컬 영상 파일")

        ttk.Label(file_frame, text="영상 파일:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.video_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="찾아보기",
                   command=self.browse_file).grid(row=0, column=2, padx=5)

        # 유튜브 탭
        youtube_frame = ttk.Frame(notebook, padding="10")
        notebook.add(youtube_frame, text="YouTube URL")

        ttk.Label(youtube_frame, text="YouTube URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(youtube_frame, textvariable=self.youtube_url, width=50).grid(row=0, column=1, columnspan=2, padx=5)

        # 설정 프레임
        settings_frame = ttk.LabelFrame(main_frame, text="추출 설정", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # 출력 디렉토리
        ttk.Label(settings_frame, text="저장 폴더:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.output_dir, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(settings_frame, text="선택",
                   command=self.browse_output_dir).grid(row=0, column=2, padx=5)

        # 유사도 임계값
        ttk.Label(settings_frame, text="유사도 임계값:").grid(row=1, column=0, sticky=tk.W, pady=5)
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)

        threshold_scale = ttk.Scale(threshold_frame, from_=0.8, to=0.99,
                                    variable=self.similarity_threshold,
                                    orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=0, padx=5)
        threshold_label = ttk.Label(threshold_frame, textvariable=self.similarity_threshold)
        threshold_label.grid(row=0, column=1, padx=5)

        # 프레임 추출 간격
        ttk.Label(settings_frame, text="추출 간격(초):").grid(row=2, column=0, sticky=tk.W, pady=5)
        interval_spin = ttk.Spinbox(settings_frame, from_=1, to=10,
                                    textvariable=self.frame_interval, width=10)
        interval_spin.grid(row=2, column=1, sticky=tk.W, padx=5)

        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)

        self.process_button = ttk.Button(button_frame, text="슬라이드 추출 시작",
                                         command=self.start_processing, style="Accent.TButton")
        self.process_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(button_frame, text="중지",
                                      command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=10)

        ttk.Button(button_frame, text="결과 폴더 열기",
                   command=self.open_output_folder).grid(row=0, column=2, padx=10)

        # 진행률 표시
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        self.progress_label = ttk.Label(progress_frame, text="대기 중...")
        self.progress_label.grid(row=0, column=0, sticky=tk.W)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # 로그 표시
        log_frame = ttk.LabelFrame(main_frame, text="처리 로그", padding="5")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 그리드 가중치 설정
        main_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)

        # 스타일 설정
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    def browse_file(self):
        """영상 파일 선택"""
        filename = filedialog.askopenfilename(
            title="영상 파일 선택",
            filetypes=[
                ("영상 파일", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("모든 파일", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)

    def browse_output_dir(self):
        """출력 디렉토리 선택"""
        dirname = filedialog.askdirectory(title="저장 폴더 선택")
        if dirname:
            self.output_dir.set(dirname)

    def log_message(self, message):
        """로그 메시지 큐에 추가"""
        self.log_queue.put(message)

    def check_queue(self):
        """로그 큐 확인하여 GUI 업데이트"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        # 100ms마다 큐 확인
        self.root.after(100, self.check_queue)

    def validate_input(self):
        """입력 유효성 검사"""
        # 로컬 파일 또는 YouTube URL 중 하나는 있어야 함
        has_local_file = bool(self.video_path.get().strip())
        has_youtube_url = bool(self.youtube_url.get().strip())

        if not has_local_file and not has_youtube_url:
            messagebox.showerror("입력 오류", "로컬 영상 파일 또는 YouTube URL을 입력해주세요.")
            return False

        if has_local_file and has_youtube_url:
            messagebox.showwarning("입력 주의", "로컬 파일과 YouTube URL 중 하나만 선택해주세요.")
            return False

        # 로컬 파일 존재 확인
        if has_local_file and not os.path.exists(self.video_path.get()):
            messagebox.showerror("파일 오류", "선택한 영상 파일이 존재하지 않습니다.")
            return False

        # 출력 디렉토리 확인
        if not self.output_dir.get().strip():
            messagebox.showerror("설정 오류", "저장 폴더를 지정해주세요.")
            return False

        return True

    def processing_worker(self):
        """백그라운드에서 실행될 처리 함수"""
        try:
            # PPTExtractor 임포트 및 초기화
            from ppt_extractor import PPTExtractor

            extractor = PPTExtractor(
                output_dir=self.output_dir.get(),
                similarity_threshold=self.similarity_threshold.get()
            )

            # 로그 메시지를 큐로 전달하도록 수정
            original_logger = extractor.logger

            class GUILogger:
                def __init__(self, log_queue):
                    self.log_queue = log_queue

                def info(self, message):
                    self.log_queue.put(f"[정보] {message}")

                def error(self, message):
                    self.log_queue.put(f"[오류] {message}")

                def warning(self, message):
                    self.log_queue.put(f"[경고] {message}")

            extractor.logger = GUILogger(self.log_queue)

            # 처리 시작
            if self.youtube_url.get().strip():
                source = self.youtube_url.get().strip()
                is_youtube = True
                self.log_message("YouTube 영상 처리 시작...")
            else:
                source = self.video_path.get().strip()
                is_youtube = False
                self.log_message("로컬 영상 파일 처리 시작...")

            # 프레임 간격 설정
            extractor.extract_frames = lambda video_path, interval=self.frame_interval.get(): \
                original_extractor_extract_frames(extractor, video_path, interval)

            success = extractor.process_video(source, is_youtube=is_youtube)

            if success:
                self.log_message("✅ 처리가 성공적으로 완료되었습니다!")
                self.root.after(0, lambda: messagebox.showinfo("완료", "슬라이드 추출이 완료되었습니다!"))
            else:
                self.log_message("❌ 처리 중 오류가 발생했습니다.")
                self.root.after(0, lambda: messagebox.showerror("오류", "처리 중 오류가 발생했습니다."))

        except Exception as e:
            error_msg = f"예상치 못한 오류: {str(e)}"
            self.log_message(error_msg)
            self.root.after(0, lambda: messagebox.showerror("오류", error_msg))

        finally:
            # UI 상태 복원
            self.root.after(0, self.processing_finished)

    def start_processing(self):
        """처리 시작"""
        if not self.validate_input():
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_label.config(text="처리 중...")
        self.progress_bar.start()

        # 로그 초기화
        self.log_text.delete(1.0, tk.END)

        # 백그라운드 스레드에서 처리 시작
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """처리 중지"""
        self.is_processing = False
        self.log_message("사용자에 의해 중지되었습니다.")
        self.processing_finished()

    def processing_finished(self):
        """처리 완료 후 UI 상태 복원"""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_label.config(text="완료")
        self.progress_bar.stop()

    def open_output_folder(self):
        """결과 폴더 열기"""
        output_path = Path(self.output_dir.get())
        if output_path.exists():
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", output_path])
            else:
                subprocess.run(["xdg-open", output_path])
        else:
            messagebox.showwarning("폴더 없음", "출력 폴더가 존재하지 않습니다.")


def original_extractor_extract_frames(extractor, video_path, interval):
    """원본 extract_frames 메서드 (interval 매개변수 지원)"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * interval

    frames = []
    frame_count = 0

    extractor.logger.info("프레임 추출 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    extractor.logger.info(f"총 {len(frames)}개 프레임 추출 완료")
    return frames


def main():
    root = tk.Tk()
    app = PPTExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()