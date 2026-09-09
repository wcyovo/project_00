// ==================== 全局状态 ====================

let currentUser = null;
let currentFrameIdx = 0;
let currentTotalFrames = 0;
let currentPosture = null;
let allPostures = [];
let postureGroups = {};  // 存储每个用户的睡姿分组信息
let allFrameLabels = []; // 存储所有帧的标签

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', function() {
    loadGlobalStats();
    loadPostures();
});

// ==================== API 调用 ====================

async function apiFetch(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return null;
    }
}

// ==================== 全局统计 ====================

async function loadGlobalStats() {
    const data = await apiFetch('/api/statistics');
    if (!data) return;
    
    document.getElementById('statTotalFrames').textContent = data.total_frames || '-';
    document.getElementById('statTotalUsers').textContent = data.total_users || '-';
    
    const postureStats = document.getElementById('postureStats');
    postureStats.innerHTML = '';
    const colors = {
        '仰卧': '#4299e1',
        '俯卧': '#ecc94b',
        '左侧卧': '#48bb78',
        '右侧卧': '#fc8181'
    };
    
    if (data.posture_counts) {
        for (const [posture, count] of Object.entries(data.posture_counts)) {
            const div = document.createElement('div');
            div.className = 'posture-stat';
            div.innerHTML = `
                <span class="color-dot" style="background:${colors[posture] || '#718096'}"></span>
                ${posture}: ${count}帧
            `;
            postureStats.appendChild(div);
        }
    }
}

async function loadPostures() {
    const data = await apiFetch('/api/postures');
    if (data) {
        allPostures = data.postures || [];
    }
}

// ==================== 用户切换 ====================

async function onUserChange() {
    const select = document.getElementById('userSelect');
    currentUser = select.value;
    
    if (!currentUser) {
        document.getElementById('userInfo').style.display = 'none';
        document.getElementById('frameControl').style.display = 'none';
        document.getElementById('postureFilter').style.display = 'none';
        document.getElementById('heatmapPlaceholder').style.display = 'block';
        document.getElementById('heatmapImage').style.display = 'none';
        document.getElementById('heatmapInfo').style.display = 'none';
        return;
    }
    
    const data = await apiFetch(`/api/user/${currentUser}`);
    if (!data) return;
    
    // 获取该用户的所有帧信息（包含分组）
    const framesData = await apiFetch(`/api/user/${currentUser}/frames`);
    if (framesData && framesData.posture_groups) {
        // 重新构建 postureGroups，确保每个睡姿的起始索引和数量正确
        postureGroups = {};
        for (const [posture, info] of Object.entries(framesData.posture_groups)) {
            postureGroups[posture] = {
                start_idx: info.start_idx,
                count: info.count
            };
        }
        allFrameLabels = framesData.frames ? framesData.frames.map(f => f.posture) : [];
        console.log('📊 睡姿分组（更新后）:', postureGroups);
        console.log('📊 帧标签数量:', allFrameLabels.length);
    } else {
        console.warn('⚠️ 未获取到 posture_groups 数据');
    }
    
    document.getElementById('userInfo').style.display = 'block';
    document.getElementById('displayUsername').textContent = data.username;
    document.getElementById('displayTotalFrames').textContent = data.total_frames;
    
    const distDiv = document.getElementById('postureDistribution');
    distDiv.innerHTML = '';
    if (data.posture_counts) {
        for (const [posture, count] of Object.entries(data.posture_counts)) {
            const tag = document.createElement('span');
            tag.className = `posture-tag ${posture}`;
            tag.textContent = `${posture}: ${count}`;
            distDiv.appendChild(tag);
        }
    }
    
    document.getElementById('frameControl').style.display = 'block';
    currentTotalFrames = data.total_frames;
    currentFrameIdx = 0;
    document.getElementById('frameSlider').max = currentTotalFrames - 1;
    document.getElementById('frameSlider').value = 0;
    updateFrameInfo();
    
    document.getElementById('postureFilter').style.display = 'block';
    const filterDiv = document.getElementById('postureFilterButtons');
    filterDiv.innerHTML = '';
    const allBtn = document.createElement('button');
    allBtn.className = 'posture-filter-btn active';
    allBtn.textContent = '全部';
    allBtn.dataset.posture = '';
    allBtn.onclick = () => filterByPosture('');
    filterDiv.appendChild(allBtn);
    
    // 按固定顺序显示
    const postureOrder = ['仰卧', '俯卧', '左侧卧', '右侧卧'];
    const availablePostures = data.postures || [];
    for (const posture of postureOrder) {
        if (availablePostures.includes(posture)) {
            const btn = document.createElement('button');
            btn.className = 'posture-filter-btn';
            btn.textContent = posture;
            btn.dataset.posture = posture;
            btn.onclick = () => filterByPosture(posture);
            filterDiv.appendChild(btn);
        }
    }
    
    currentPosture = null;
    loadHeatmap(0);
}

// ==================== 热力图加载 ====================

async function loadHeatmap(idx) {
    if (!currentUser) return;
    
    console.log(`📊 loadHeatmap 被调用, idx=${idx}`);
    
    let url = `/api/user/${currentUser}/heatmap?idx=${idx}`;
    if (currentPosture) {
        url += `&posture=${encodeURIComponent(currentPosture)}`;
    }
    
    const data = await apiFetch(url);
    if (!data) return;
    
    console.log(`📊 后端返回: frame_idx=${data.frame_idx}, posture=${data.posture}`);
    
    const img = document.getElementById('heatmapImage');
    img.src = data.image;
    img.style.display = 'block';
    document.getElementById('heatmapPlaceholder').style.display = 'none';
    document.getElementById('heatmapInfo').style.display = 'flex';
    
    document.getElementById('heatmapUser').textContent = `👤 ${data.username}`;
    document.getElementById('heatmapPosture').textContent = `🏷️ ${data.posture}`;
    document.getElementById('heatmapFrame').textContent = `帧 ${data.frame_idx + 1} / ${data.total_frames}`;
    
    // 更新全局状态
    currentFrameIdx = data.frame_idx;
    document.getElementById('frameSlider').value = currentFrameIdx;
    updateFrameInfo();
}

// ==================== 帧控制 ====================

function updateFrameInfo() {
    console.log('🔄 updateFrameInfo 被调用');
    console.log('   currentPosture:', currentPosture);
    console.log('   postureGroups:', postureGroups);
    
    if (currentPosture && postureGroups[currentPosture]) {
        const group = postureGroups[currentPosture];
        const relativeIdx = currentFrameIdx - group.start_idx;
        document.getElementById('frameInfo').textContent = `第${relativeIdx + 1}帧 / ${group.count}帧`;
        console.log(`   显示: 第${relativeIdx + 1}帧 / ${group.count}帧`);
    } else if (currentPosture) {
        console.warn(`⚠️ 当前睡姿 ${currentPosture} 不在 postureGroups 中`);
        document.getElementById('frameInfo').textContent = `第${currentFrameIdx + 1}帧 / ${currentTotalFrames}帧`;
    } else {
        document.getElementById('frameInfo').textContent = `第${currentFrameIdx + 1}帧 / ${currentTotalFrames}帧`;
    }
}

function nextFrame() {
    console.log('🔄 nextFrame 被调用');
    console.log('   currentPosture:', currentPosture);
    console.log('   currentFrameIdx:', currentFrameIdx);
    console.log('   postureGroups:', postureGroups);
    
    if (currentPosture && postureGroups[currentPosture]) {
        const group = postureGroups[currentPosture];
        const relativeIdx = currentFrameIdx - group.start_idx;
        console.log(`   相对索引: ${relativeIdx}, 总数: ${group.count}`);
        
        if (relativeIdx < group.count - 1) {
            const newIdx = currentFrameIdx + 1;
            console.log(`   跳转到帧 ${newIdx}`);
            currentFrameIdx = newIdx;
            loadHeatmap(currentFrameIdx);
        } else {
            console.log('⚠️ 已是最后一帧');
        }
    } else if (currentPosture) {
        console.warn(`⚠️ 当前睡姿 ${currentPosture} 不在 postureGroups 中`);
    } else {
        if (currentFrameIdx < currentTotalFrames - 1) {
            currentFrameIdx++;
            loadHeatmap(currentFrameIdx);
        }
    }
}

function prevFrame() {
    console.log('🔄 prevFrame 被调用');
    
    if (currentPosture && postureGroups[currentPosture]) {
        const group = postureGroups[currentPosture];
        const relativeIdx = currentFrameIdx - group.start_idx;
        
        if (relativeIdx > 0) {
            currentFrameIdx--;
            loadHeatmap(currentFrameIdx);
        }
    } else if (currentPosture) {
        console.warn(`⚠️ 当前睡姿 ${currentPosture} 不在 postureGroups 中`);
    } else {
        if (currentFrameIdx > 0) {
            currentFrameIdx--;
            loadHeatmap(currentFrameIdx);
        }
    }
}

function onSliderChange() {
    const val = parseInt(document.getElementById('frameSlider').value);
    currentFrameIdx = val;
    loadHeatmap(currentFrameIdx);
}

// ==================== 睡姿筛选 ====================

function filterByPosture(posture) {
    console.log('🔄 filterByPosture 被调用, posture:', posture);
    console.log('📊 当前 postureGroups:', postureGroups);
    
    currentPosture = posture;
    
    document.querySelectorAll('.posture-filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.posture === posture);
    });
    
    if (posture && postureGroups[posture]) {
        // 跳转到该睡姿的第一帧
        const startIdx = postureGroups[posture].start_idx;
        console.log(`📊 切换到 ${posture}, 起始索引 ${startIdx}, 总数 ${postureGroups[posture].count}`);
        currentFrameIdx = startIdx;
        loadHeatmap(currentFrameIdx);
    } else if (posture) {
        console.warn(`⚠️ 未找到睡姿 ${posture} 的分组信息`);
        currentFrameIdx = 0;
        loadHeatmap(currentFrameIdx);
    } else {
        currentFrameIdx = 0;
        loadHeatmap(currentFrameIdx);
    }
}

// ==================== 键盘快捷键 ====================

document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        nextFrame();
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevFrame();
    }
});