# Nhận Diện Côn Trùng trên Bẫy Dính Màu Vàng

Dự án này cung cấp một quy trình đầy đủ để phát hiện và phân loại côn trùng trong hình ảnh bẫy dính màu vàng sử dụng YOLOv11. Hệ thống sử dụng phương pháp phân chia ảnh (tiling) để xử lý hiệu quả hình ảnh có độ phân giải cao.

> **Ghi chú**: Để tái tạo nhanh kết quả, bạn có thể sử dụng file `training-inference.ipynb`.

## Tính năng

- **Xử lý Dữ liệu**: Phân chia hình ảnh độ phân giải cao với độ chồng lấp để phát hiện tốt hơn côn trùng ở vùng biên
- **Huấn luyện Mô hình**: Huấn luyện mô hình YOLOv11 với tham số tối ưu cho việc phát hiện côn trùng nhỏ
- **Suy luận**: Xử lý hình ảnh mới với phương pháp phân chia và áp dụng Non-Maximum Suppression để kết hợp kết quả
- **Phân tích Dữ liệu Khám phá**: Công cụ phân tích dữ liệu và hiển thị các thống kê

## Các loại côn trùng

- WF: Bọ phấn trắng (Whitefly)
- MR: Rệp sáp (Mealybug)
- NC: Nhện (Mite)

## Cấu trúc dự án

```
Yellow Sticky Trap/
├── config/                   # File cấu hình
│   └── default.yaml          # Cấu hình mặc định
├── notebooks/                # Jupyter notebooks để khám phá
├── src/                      # Mã nguồn
│   ├── data/                 # Module xử lý dữ liệu
│   │   └── preprocessing.py  # Phân chia và tiền xử lý hình ảnh
│   ├── models/               # Mã liên quan đến mô hình
│   │   ├── train.py          # Hàm huấn luyện
│   │   └── inference.py      # Hàm suy luận
│   └── utils/                # Tiện ích
│       └── eda.py            # Công cụ phân tích dữ liệu khám phá
├── main.py                   # Điểm vào CLI chính
├── yolo-nn.ipynb             # Notebook gốc để tái tạo nhanh kết quả
├── requirements.txt          # Các thư viện phụ thuộc
├── LICENSE                   # File giấy phép
└── README.md                 # Tài liệu dự án
```

## Cài đặt

1. Clone repository này:
   ```
   git clone
   ```

2. Cài đặt các thư viện phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

## Sử dụng

### Sử dụng Notebook

Cách đơn giản nhất để chạy toàn bộ quy trình từ tiền xử lý đến huấn luyện và dự đoán là sử dụng notebook `training-inference.ipynb`:

1. Mở notebook trong Jupyter hoặc Colab
2. Chạy các ô theo thứ tự để xử lý dữ liệu, huấn luyện và dự đoán

### Sử dụng Command Line

#### Chuẩn bị Dataset

Để chuẩn bị dataset bằng cách phân chia hình ảnh:

```bash
python main.py prepare --img-dir /đường/dẫn/ảnh --ann-dir /đường/dẫn/annotations --output-dir /đường/dẫn/đầu/ra --config config/default.yaml
```

#### Phân tích dữ liệu

Để chạy phân tích dữ liệu khám phá trên dataset:

```bash
python main.py eda --img-dir /đường/dẫn/ảnh --ann-dir /đường/dẫn/annotations
```

#### Huấn luyện mô hình

Để huấn luyện mô hình YOLOv11:

```bash
python main.py train --data /đường/dẫn/đến/insects.yaml --output runs/my_model --epochs 30
```

#### Đánh giá mô hình

Để đánh giá một mô hình đã huấn luyện:

```bash
python main.py evaluate --model /đường/dẫn/đến/best.pt --data /đường/dẫn/đến/data.yaml
```

#### Suy luận

Để chạy suy luận trên một hình ảnh mới:

```bash
python main.py infer --model /đường/dẫn/đến/best.pt --image /đường/dẫn/đến/ảnh.jpg --output /đường/dẫn/đến/kết_quả.jpg
```

## Giấy phép

Dự án này được cấp phép theo giấy phép MIT - xem file [LICENSE](LICENSE) để biết chi tiết.


