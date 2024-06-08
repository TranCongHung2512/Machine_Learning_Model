import streamlit as st

st.write("""
# :adult: Thành viên

### Nhóm gồm các thành viên:
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("images/Nam.jpg", caption="Võ Văn Nam")
    st.write("""
    - Họ tên: Võ Văn Nam
    - Mã sinh viên: 21IT683
    - Lớp sinh hoạt: 21SE3
    - Khoa: Khoa Học Máy Tính
    - Ngành: Công Nghệ Thông Tin
    """)

with col2:
    st.image("images/Linh.jpg", caption="Ngô Nguyễn Viết Lĩnh")
    st.write("""
        - Họ tên: Ngô Nguyễn Viết Lĩnh
        - Mã sinh viên: 21IT150
        - Lớp sinh hoạt: 21SE3
        - Khoa: Khoa Học Máy Tính
        - Ngành: Công Nghệ Thông Tin
    """)
