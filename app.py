import streamlit as st
import os

# 1. 페이지 설정 (웹 브라우저 탭에 표시될 정보)
st.set_page_config(
    page_title="달콤한 디저트 공방", 
    page_icon="🍰", 
    layout="centered"
)

# 2. 디자인을 위한 커스텀 CSS (폰트 크기 및 스타일)
st.markdown("""
    <style>
    .main-title {
        font-size: 40px !important;
        color: #FF69B4;
        text-align: center;
        font-weight: bold;
    }
    .dessert-box {
        border: 5px solid #FFB6C1;
        border-radius: 20px;
        padding: 20px;
        background-color: #FFF0F5;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🎨 나만의 달콤한 디저트 만들기</p>', unsafe_allow_html=True)

# 3. 사이드바: 디저트 및 색상 선택
st.sidebar.header("🛠️ 꾸미기 도구함")

# 디저트 목록 (파일 이름과 정확히 일치해야 함)
dessert_options = {
    "아이스크림": "icecream.png",
    "빙수": "shaved_ice.png",
    "젤리빈": "jellybean.png",
    "마카롱": "macaron.png",
    "쿠키": "cookie.png"
}

selected_name = st.sidebar.selectbox("어떤 디저트를 고를까요?", list(dessert_options.keys()))
chosen_color = st.sidebar.color_picker("가장 좋아하는 색깔을 골라보세요!", "#FFB6C1")

# 4. 메인 화면 로직
st.divider()

# 선택된 이미지 경로 파악
image_filename = dessert_options[selected_name]
image_path = os.path.join("assets", image_filename)

# 화면 레이아웃 구성
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"✨ 완성된 {selected_name} ✨")
    
    # 이미지 출력 시도
    if os.path.exists(image_path):
        # 이미지를 감싸는 예쁜 박스 효과
        st.image(image_path, use_column_width=True)
    else:
        st.error(f"⚠️ '{image_filename}' 파일을 찾을 수 없어요!")
        st.info(f"GitHub의 assets 폴더에 {image_filename} 파일이 있는지 다시 확인해주세요.")

with col2:
    st.write("### 📝 제작 노트")
    st.info(f"오늘의 테마: **{selected_name}**")
    
    # 선택한 색상을 시각적으로 보여주는 원형 박스
    st.write("지정된 마법 색상:")
    st.markdown(f"""
        <div style="
            width: 80px; 
            height: 80px; 
            background-color: {chosen_color}; 
            border-radius: 50%; 
            border: 3px solid white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        "></div>
    """, unsafe_allow_html=True)
    st.write(f"색상 코드: `{chosen_color}`")

# 5. 인터랙션 버튼
if st.button("🎉 디저트 가게에 진열하기"):
    st.balloons()
    st.confetti() # Streamlit의 즐거운 효과
    st.success(f"정말 멋져요! 이 {selected_name}은 세상에서 가장 달콤할 것 같아요.")
