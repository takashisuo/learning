const questions = [
  {
    text: '日本で最も高い山はどれですか？',
    choices: ['北岳', '槍ヶ岳', '富士山', '奥穂高岳'],
    correctIndex: 2,
    explanation: '正解は「富士山」（3,776m）です。日本一の高さを誇り、世界文化遺産にも登録されています。'
  },
  {
    text: '人間の体の中で最も大きな臓器はどれですか？',
    choices: ['心臓', '皮膚', '肝臓', '肺'],
    correctIndex: 1,
    explanation: '正解は「皮膚」です。体全体を覆う皮膚は成人で約1.6㎡、重さは約3kgにもなる最大の臓器です。'
  },
  {
    text: '世界で最も広い海洋はどれですか？',
    choices: ['大西洋', '北極海', 'インド洋', '太平洋'],
    correctIndex: 3,
    explanation: '正解は「太平洋」です。地球の表面積の約3分の1を占める世界最大の海洋です。'
  },
  {
    text: '「モナ・リザ」を描いた画家は誰ですか？',
    choices: ['ミケランジェロ', 'ボッティチェリ', 'ラファエロ', 'レオナルド・ダ・ヴィンチ'],
    correctIndex: 3,
    explanation: '正解は「レオナルド・ダ・ヴィンチ」です。15〜16世紀のイタリアの芸術家・科学者で、ルネサンスを代表する万能の天才です。'
  },
  {
    text: '日本の国鳥はどれですか？',
    choices: ['ツバメ', 'タンチョウ', 'キジ', 'サギ'],
    correctIndex: 2,
    explanation: '正解は「キジ」です。1947年に日本鳥類保護連盟が選定し、日本の国鳥として広く知られています。'
  }
];

let currentIndex = 0;
let score = 0;
let answered = false;

const questionNumberEl = document.getElementById('question-number');
const progressFillEl = document.getElementById('progress-fill');
const questionTextEl = document.getElementById('question-text');
const choicesEl = document.getElementById('choices');
const feedbackEl = document.getElementById('feedback');
const nextBtn = document.getElementById('next-btn');
const quizScreen = document.getElementById('quiz-screen');
const resultScreen = document.getElementById('result-screen');
const scoreTextEl = document.getElementById('score-text');
const scoreCommentEl = document.getElementById('score-comment');
const resultIconEl = document.getElementById('result-icon');
const retryBtn = document.getElementById('retry-btn');

function loadQuestion() {
  answered = false;
  const q = questions[currentIndex];

  questionNumberEl.textContent = `問題 ${currentIndex + 1} / ${questions.length}`;
  progressFillEl.style.width = `${(currentIndex / questions.length) * 100}%`;

  questionTextEl.textContent = q.text;

  choicesEl.innerHTML = '';
  q.choices.forEach((choice, i) => {
    const btn = document.createElement('button');
    btn.textContent = choice;
    btn.className = 'choice-btn';
    btn.addEventListener('click', () => selectAnswer(i));
    choicesEl.appendChild(btn);
  });

  feedbackEl.className = 'feedback hidden';
  feedbackEl.textContent = '';
  nextBtn.classList.add('hidden');
}

function selectAnswer(selectedIndex) {
  if (answered) return;
  answered = true;

  const q = questions[currentIndex];
  const choiceBtns = choicesEl.querySelectorAll('.choice-btn');

  choiceBtns.forEach(btn => btn.disabled = true);

  if (selectedIndex === q.correctIndex) {
    score++;
    choiceBtns[selectedIndex].classList.add('correct');
    feedbackEl.textContent = '正解！ ' + q.explanation;
    feedbackEl.className = 'feedback correct';
  } else {
    choiceBtns[selectedIndex].classList.add('incorrect');
    choiceBtns[q.correctIndex].classList.add('highlight');
    feedbackEl.textContent = '不正解… ' + q.explanation;
    feedbackEl.className = 'feedback incorrect';
  }

  const isLast = currentIndex === questions.length - 1;
  nextBtn.textContent = isLast ? '結果を見る' : '次の問題へ →';
  nextBtn.classList.remove('hidden');
}

function showResult() {
  quizScreen.classList.add('hidden');
  resultScreen.classList.remove('hidden');

  scoreTextEl.textContent = `${questions.length}問中 ${score}問正解`;

  if (score === questions.length) {
    resultIconEl.textContent = '🏆';
    scoreCommentEl.textContent = '全問正解！素晴らしい！';
  } else if (score >= 3) {
    resultIconEl.textContent = '👍';
    scoreCommentEl.textContent = 'よく頑張りました！';
  } else {
    resultIconEl.textContent = '📚';
    scoreCommentEl.textContent = 'もう一度挑戦してみましょう！';
  }
}

nextBtn.addEventListener('click', () => {
  currentIndex++;
  if (currentIndex < questions.length) {
    loadQuestion();
  } else {
    progressFillEl.style.width = '100%';
    showResult();
  }
});

retryBtn.addEventListener('click', () => {
  currentIndex = 0;
  score = 0;
  resultScreen.classList.add('hidden');
  quizScreen.classList.remove('hidden');
  loadQuestion();
});

loadQuestion();
