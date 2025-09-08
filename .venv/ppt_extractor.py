import cv2
import numpy as np
import os
import yt_dlp
from PIL import Image
import img2pdf
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
from pathlib import Path
import logging


class PPTExtractor:
    def __init__(self, output_dir="extracted_slides", similarity_threshold=0.95):
        """
        PPT 슬라이드 추출기 초기화

        Args:
            output_dir: 슬라이드 저장 디렉토리
            similarity_threshold: 이미지 유사도 임계값 (0-1)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.extracted_slides = []

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_youtube_video(self, url, temp_dir="temp"):
        """유튜브 영상 다운로드"""
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)

        ydl_opts = {
            'format': 'best[height<=720]',  # 720p 이하 품질
            'outtmpl': f'{temp_dir}/%(title)s.%(ext)s',
            'quiet': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                self.logger.info(f"영상 다운로드 완료: {filename}")
                return filename
        except Exception as e:
            self.logger.error(f"영상 다운로드 실패: {e}")
            return None

    def extract_frames(self, video_path, interval=2):
        """
        영상에서 프레임 추출

        Args:
            video_path: 영상 파일 경로
            interval: 프레임 추출 간격 (초)
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps * interval

        frames = []
        frame_count = 0

        self.logger.info("프레임 추출 시작...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append(frame)

            frame_count += 1

        cap.release()
        self.logger.info(f"총 {len(frames)}개 프레임 추출 완료")
        return frames

    def calculate_image_features(self, image):
        """이미지 특징 벡터 계산"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 히스토그램 계산
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / np.sum(hist)  # 정규화

        # 에지 특징
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 특징 벡터 결합
        features = np.concatenate([hist, [edge_density]])
        return features

    def is_similar_slide(self, img1, img2):
        """두 이미지가 유사한 슬라이드인지 판단"""
        # 이미지 크기 조정
        img1_resized = cv2.resize(img1, (640, 480))
        img2_resized = cv2.resize(img2, (640, 480))

        # 특징 벡터 계산
        features1 = self.calculate_image_features(img1_resized)
        features2 = self.calculate_image_features(img2_resized)

        # 코사인 유사도 계산
        similarity = cosine_similarity([features1], [features2])[0][0]

        return similarity > self.similarity_threshold

    def remove_duplicate_slides(self, frames):
        """중복 슬라이드 제거"""
        if not frames:
            return []

        unique_slides = [frames[0]]
        self.logger.info("중복 슬라이드 제거 중...")

        for i, current_frame in enumerate(frames[1:], 1):
            is_duplicate = False

            # 최근 몇 개 슬라이드와만 비교 (성능 최적화)
            comparison_range = min(5, len(unique_slides))
            for j in range(comparison_range):
                if self.is_similar_slide(current_frame, unique_slides[-(j + 1)]):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_slides.append(current_frame)
                self.logger.info(f"새로운 슬라이드 발견: {len(unique_slides)}번째")

        self.logger.info(f"중복 제거 완료: {len(unique_slides)}개 고유 슬라이드")
        return unique_slides

    def save_slides(self, slides, prefix="slide"):
        """슬라이드 이미지 저장"""
        saved_paths = []

        for i, slide in enumerate(slides, 1):
            filename = f"{prefix}_{i:03d}.png"
            filepath = self.output_dir / filename

            cv2.imwrite(str(filepath), slide)
            saved_paths.append(filepath)
            self.logger.info(f"슬라이드 저장: {filename}")

        self.extracted_slides = saved_paths
        return saved_paths

    def create_pdf(self, image_paths, pdf_name="extracted_slides.pdf"):
        """이미지들을 PDF로 변환"""
        if not image_paths:
            self.logger.warning("PDF로 변환할 이미지가 없습니다.")
            return None

        try:
            # PIL Image로 변환하여 크기 통일
            processed_images = []
            for img_path in image_paths:
                img = Image.open(img_path)
                # A4 비율로 조정
                img = img.convert('RGB')
                processed_images.append(img_path)

            pdf_path = self.output_dir / pdf_name

            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(processed_images))

            self.logger.info(f"PDF 생성 완료: {pdf_path}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"PDF 생성 실패: {e}")
            return None

    def process_video(self, video_source, is_youtube=False):
        """
        메인 처리 함수

        Args:
            video_source: 영상 경로 또는 YouTube URL
            is_youtube: YouTube 링크 여부
        """
        try:
            # 1. 영상 준비
            if is_youtube:
                video_path = self.download_youtube_video(video_source)
                if not video_path:
                    return False
            else:
                video_path = video_source
                if not os.path.exists(video_path):
                    self.logger.error(f"영상 파일을 찾을 수 없습니다: {video_path}")
                    return False

            # 2. 프레임 추출
            frames = self.extract_frames(video_path, interval=3)
            if not frames:
                self.logger.error("프레임 추출에 실패했습니다.")
                return False

            # 3. 중복 제거
            unique_slides = self.remove_duplicate_slides(frames)

            # 4. 슬라이드 저장
            saved_paths = self.save_slides(unique_slides)

            # 5. PDF 생성
            pdf_path = self.create_pdf(saved_paths)

            self.logger.info("=" * 50)
            self.logger.info("처리 완료!")
            self.logger.info(f"추출된 슬라이드 수: {len(saved_paths)}")
            self.logger.info(f"저장 위치: {self.output_dir}")
            if pdf_path:
                self.logger.info(f"PDF 파일: {pdf_path}")
            self.logger.info("=" * 50)

            # 임시 파일 정리 (YouTube 다운로드인 경우)
            if is_youtube and os.path.exists(video_path):
                os.remove(video_path)
                self.logger.info("임시 파일 삭제 완료")

            return True

        except Exception as e:
            self.logger.error(f"처리 중 오류 발생: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="PPT 슬라이드 추출 툴")
    parser.add_argument("source", help="영상 파일 경로 또는 YouTube URL")
    parser.add_argument("--output", "-o", default="extracted_slides",
                        help="출력 디렉토리 (기본값: extracted_slides)")
    parser.add_argument("--threshold", "-t", type=float, default=0.95,
                        help="유사도 임계값 (기본값: 0.95)")
    parser.add_argument("--youtube", "-y", action="store_true",
                        help="YouTube URL 처리")

    args = parser.parse_args()

    # 추출기 초기화
    extractor = PPTExtractor(
        output_dir=args.output,
        similarity_threshold=args.threshold
    )

    # 처리 실행
    success = extractor.process_video(args.source, is_youtube=args.youtube)

    if success:
        print("✅ 슬라이드 추출이 완료되었습니다!")
    else:
        print("❌ 슬라이드 추출에 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()