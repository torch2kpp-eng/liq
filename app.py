import streamlit as st
from PIL import Image, ImageChops
import numpy as np

# 페이지 레이아웃 설정
st.set_page_config(page_title="디저트 아틀리에", layout="wide")

def apply_color(base_image_path, mask_image_path, color_hex):
    """
    베이스 이미지에 사용자가 선택한 색상을 합성하는 함수
    """
    base = Image.open(base_image_path).convert("RGBA")
    mask = Image.open(mask_image_path).convert("RGBA")
    
    # Hex 색상을 RGB로 변환
    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # 선택한 색상의 단색 이미지 생성
    color_layer = Image.new("RGBA", base.size, color_rgb + (255,))
    
    # 마스크를 이용하여 특정 영역에만 색상 적용
    colored_part = ImageChops.multiply(color_layer, mask)
    
    # 최종 합성 (베이스 이미지 위에 색상이 입혀진 레이어를 얹음)
    final_image = Image.alpha_composite(base, colored_part)
    return final_image

# UI 구성
st.title("👨‍🎨 디지털 디저트 아틀리에")
st.sidebar.header("🎨 디자인 설정")

# 사용자 입력
dessert_choice = st.sidebar.selectbox("디저트 종류", ["컵케이크", "푸딩", "도넛"])
selected_color = st.sidebar.color_picker("원하는 색상을 선택하세요", "#FF4B4B")
topping_on = st.sidebar.checkbox("스프링클 추가")

# 메인 렌더링 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("작업대")
    try:
        # 실제 환경에서는 파일 경로를 디저트 종류에 따라 매핑
        # 예: base_path = f"assets/{dessert_choice}_base.png"
        
        # 데모를 위한 처리 (이미지가 없을 경우를 대비한 로직)
        result_img = apply_color("cupcake_base.png", "cupcake_mask.png", selected_color)
        
        if topping_on:
            # 토핑 이미지가 있다면 추가로 합성 가능
            topping_img = Image.open("sprinkles.png").convert("RGBA")
            result_img = Image.alpha_composite(result_img, topping_img)
            
        st.image(result_img, caption=f"당신만의 {dessert_choice}", use_column_width=True)
        
    except FileNotFoundError:
        st.warning("이미지 파일(base/mask)이 필요합니다. 프로젝트 폴더에 PNG 파일을 준비해주세요.")
        st.info("Tip: 마스크 이미지는 색을 칠할 영역만 흰색인 이미지여야 합니다.")

with col2:
    st.subheader("레시피 정보")
    st.write(f"**선택된 베이스:** {dessert_choice}")
    st.write(f"**적용된 컬러 코드:** `{selected_color}`")
    
    # 이미지 다운로드 기능
    if 'result_img' in locals():
        import io
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="🖼️ 이미지 저장하기", data=byte_im, file_name="my_dessert.png", mime="image/png")

