# Frame Search (Orange Car + Red Carpet)

Yêu cầu: Tìm các keyframe có (1) xe màu cam và (2) thảm đỏ, trả về frame index và thông tin khung hình.

## Cấu trúc dữ liệu kỳ vọng

- `Keyframes/`: keyframe theo thư mục tên video, ví dụ `Keyframes/L01_V001/0000.jpg`
- `Objects/`: JSON object detection tương ứng, ví dụ `Objects/L01_V001/0000.json`
- `Metadata/`: JSON metadata mỗi video, ví dụ `Metadata/L01_V001.json`
- `Map_Keyframes/`: CSV chứa mapping `frame`, `giây/second`, `fps`, `frame_index`
- (Tuỳ chọn) `CLIP/` chứa `.npy` features toàn bộ frames nếu muốn mở rộng truy vấn semantic.

## Cài đặt

```bash
python -m pip install -r app/requirements.txt
```

## Chạy tìm kiếm (CLI)

```bash
python -m app.search_frames \
  --keyframes_dir /path/to/Keyframes \
  --objects_dir /path/to/Objects \
  --metadata_dir /path/to/Metadata \
  --map_keyframes_csv /path/to/Map_Keyframes/mapping.csv \
  --output_dir /workspace/output \
  --num_results 50
```

- Mặc định dùng heuristic màu (HSV) + object detection để đảm bảo nhanh và ổn định (không cần CLIP).
- Có thể thêm `--enable_clip_text_filter` nếu đã tự cài `open_clip_torch`/`torch` để lọc thêm theo ngữ nghĩa.

## Kết quả

- In ra danh sách frame khớp điều kiện, gồm: `video`, `frame_path`, `frame_index`, `time_sec`, `fps`, `scores`.
- Lưu thumbnail có rectangle vùng ô tô sang `output_dir`.

## Ghi chú hiệu năng

- Quét `Objects/` để lọc khung hình có ô tô trước, sau đó mới đọc ảnh để kiểm tra màu cam và thảm đỏ ở vùng cần.
- Đọc ảnh lười (chỉ khi cần), xử lý theo batch nhỏ để giảm IO.