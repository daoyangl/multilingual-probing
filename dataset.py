from utils import add_noise
import pandas as pd
import json
from datasets import load_dataset

class DataProcessing():
    
    def __init__(self, language, data_path, data_name, noise):
        self.language = language
        self.data_path = data_path
        self.data_name = data_name
        self.noise = noise
    
    """
        Select the way to get p and n samples given the dataset.
    """
    def dispacher(self):
        cot = None
        if self.data_name == 'StrategyQA':
            p, q = self.StrategyQA()
            prompt = 'Judge the question is true or false?'  + '\n' \
                'Q: Will Queen Elizabeth be buried in the Pantheon?' + '\n' \
                "Let us think step by step. The stem of the sentence is Queen Elizabeth, burial, pantheon. Inference: First, the Pantheon is a church, so it is possible that she could be buried there. Second, Queen Elizabeth II is still alive, so she has not been buried yet. Third, even if she were to be buried in the Pantheon, it is unlikely that we would know about it ahead of time, so it is hard to say for sure." + '\n' \
                'pred_ans: no'
            cot = 'Let us think step by step...'
            
        elif self.data_name == 'coinflip':
            p, q, prompt = self.coinflip()

        elif self.data_name == 'cities':
            p, q, prompt = self.cities()
            
        elif self.data_name == 'common':
            p, q, prompt = self.common()

        elif self.data_name == 'counterfact':
            p, q, prompt = self.counterfact()

        elif self.data_name == 'hateeval':
            p, q = self.hateeval()
            prompt = 'According to the comment, tell whether they present hate speech or not.'
            
        elif self.data_name == 'STSA':
            p, q, prompt = self.STSA()

        elif self.data_name == 'IMDb':
            p, q = self.IMDb()
            prompt = 'According to the movie review, judge whether it is Positive or Negative.'
            
        elif self.data_name == 'sarcasm':
            p, q, prompt, cot = self.sarcasm()

        elif self.data_name == 'awareness':
            p, q, prompt = self.awareness()

        elif self.data_name == 'ethics':
            p, q, prompt = self.ethics()

        elif self.data_name == 'external_data':
            p, q, prompt = self.external_data()

        elif self.data_name == 'finance':
            p, q, prompt = self.finance()

        elif self.data_name == 'politifact':
            p, q, prompt = self.politifact()

        elif self.data_name == 'sarcasm_new':
            p, q, prompt, cot = self.sarcasm_new()

        elif self.data_name == 'airline':
            p, q, prompt = self.airline()

        elif self.data_name == 'opinion':
            p, q, prompt = self.opinion()

        return p, q, prompt, cot
    
    """
        return the context that we enter to LLM.
    """
    def get_prompt(self, prompt, cot, question):
        if self.data_name in ['common','cities','counterfact','awareness','ethics','external_data','finance','politifact','airline','opinion']:
            new_prompt = prompt + " " + question
        elif self.data_name in ['STSA','coinflip','IMDb','hateeval']:
            new_prompt = question + " " + prompt
        elif self.data_name in ['sarcasm','StrategyQA','sarcasm_new']:
            new_prompt = prompt + question + " " + cot
        
        return new_prompt

    def opinion(self):
        csv_file_path = './dataset/' + self.language + '/opinion.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['polarity'] == 'positive']['text'].tolist()
        list_false = data[data['polarity'] == 'negative']['text'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Judge the statement is Positive or Negative.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是正面的还是负面的。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est positif ou négatif."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage positiv oder negativ ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut positif atau negatif.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es positiva o negativa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸಕಾರಾತ್ಮಕ ಅಥವಾ ಋಣಾತ್ಮಕ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің оң немесе теріс екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସକାରାତ୍ମକ କି ନକାରାତ୍ମକ।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň oňyn ýa-da otrisatelidigini kesgitläň.'
        elif self.language == 'Russian':
            prompt = 'Оцените, является ли утверждение положительным или отрицательным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सकारात्मक है या नकारात्मक।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်အနုမြူရောအနုတ်မြူရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay positibo o negatibo.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he maikaʻi a i ʻole he maikaʻi ʻole ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'வாக்குமூலம் நேர்மையானதா அல்லது எதிர்மையானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి సానుకూలం లేదా ప్రతికూలం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin olumlu veya olumsuz olduğunu değerlendirin.'

        return list_true_noise, list_false_noise, prompt

    def airline(self):
        csv_file_path = './dataset/' + self.language + '/airline.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['airline_sentiment'] == 'positive']['text'].tolist()
        list_false = data[data['airline_sentiment'] == 'negative']['text'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Judge the statement is Positive or Negative.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是正面的还是负面的。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est positif ou négatif."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage positiv oder negativ ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut positif atau negatif.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es positiva o negativa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸಕಾರಾತ್ಮಕ ಅಥವಾ ಋಣಾತ್ಮಕ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің оң немесе теріс екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସକାରାତ୍ମକ କି ନକାରାତ୍ମକ।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň oňyn ýa-da otrisatelidigini kesgitläň.'
        elif self.language == 'Russian':
            prompt = 'Оцените, является ли утверждение положительным или отрицательным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सकारात्मक है या नकारात्मक।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်အနုမြူရောအနုတ်မြူရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay positibo o negatibo.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he maikaʻi a i ʻole he maikaʻi ʻole ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'வாக்குமூலம் நேர்மையானதா அல்லது எதிர்மையானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి సానుకూలం లేదా ప్రతికూలం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin olumlu veya olumsuz olduğunu değerlendirin.'

        return list_true_noise, list_false_noise, prompt

    def sarcasm_new(self):
        csv_file_path = './dataset/' + self.language + '/sarcasm.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['is_sarcastic'] == 1]['headline'].tolist()
        list_false = data[data['is_sarcastic'] == 0]['headline'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Task: Detect sarcasm. First, we need to understand what sarcasm is. Sarcasm is a form of verbal irony, where the intended meaning of the words is the opposite of the literal meaning. In other words, the speaker is saying one thing but meaning the opposite.'
            cot = 'Think carefully according to the sentence. Is there any sarcasm in this sentence? Please answer Yes or No.'
        elif self.language == 'Chinese':
            prompt = '任务：检测讽刺。首先，我们需要了解什么是讽刺。讽刺是一种言语反讽，意思是说话者表达的字面意思与实际想表达的意思相反。换句话说，说话者表面上说了一件事，但实际上表达的是相反的意思。'
            cot = '根据句子仔细思考。这个句子中是否有讽刺？请回答是或否。'
        elif self.language == 'French':
            prompt = 'Tâche : Détecter le sarcasme. Tout d\'abord, nous devons comprendre ce qu\'est le sarcasme. Le sarcasme est une forme d\'ironie verbale, où le sens des mots est à l\'opposé de leur signification littérale. En d\'autres termes, l\'interlocuteur dit une chose mais en pense une autre.'
            cot = 'Réfléchissez attentivement en fonction de la phrase. Y a-t-il du sarcasme dans cette phrase ? Veuillez répondre Oui ou Non.'
        elif self.language == 'German':
            prompt = 'Aufgabe: Sarkasmus erkennen. Zuerst müssen wir verstehen, was Sarkasmus ist. Sarkasmus ist eine Form der verbalen Ironie, bei der die beabsichtigte Bedeutung der Worte das Gegenteil der wörtlichen Bedeutung ist. Mit anderen Worten, der Sprecher sagt eine Sache, meint aber das Gegenteil.'
            cot = 'Denken Sie sorgfältig über den Satz nach. Gibt es in diesem Satz Sarkasmus? Bitte antworten Sie mit Ja oder Nein.'
        elif self.language == 'Indonesian':
            prompt = 'Tugas: Deteksi sarkasme. Pertama, kita perlu memahami apa itu sarkasme. Sarkasme adalah bentuk ironi verbal, di mana arti kata yang dimaksudkan berlawanan dengan arti literalnya. Dengan kata lain, pembicara mengatakan satu hal tetapi memiliki makna yang berlawanan.'
            cot = 'Pikirkan dengan hati-hati menurut kalimat. Apakah ada sarkasme dalam kalimat ini? Tolong jawab Ya atau Tidak.'
        elif self.language == 'Spanish':
            prompt = 'Tarea: Detectar sarcasmo. Primero, necesitamos entender qué es el sarcasmo. El sarcasmo es una forma de ironía verbal, donde el significado pretendido de las palabras es lo opuesto a su significado literal. En otras palabras, el hablante dice una cosa pero quiere decir lo contrario.'
            cot = 'Piense cuidadosamente según la frase. ¿Hay sarcasmo en esta frase? Por favor responda Sí o No.'
        elif self.language == 'Kannada':
            prompt = 'ಕಾರ್ಯ: ವಾಂಶಿಕತೆಯನ್ನು ಕಂಡುಹಿಡಿಯಿರಿ. ಮೊದಲನೆಯದಾಗಿ, ವ್ಯಂಗ್ಯವಾಕ್ಯವೆಂದರೇನು ಎಂಬುದನ್ನು ನಾವು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಬೇಕು. ವ್ಯಂಗ್ಯವಾಕ್ಯವು ಮಾತಿನ ಸಾಂದರ್ಭಿಕತೆಯ ಒಂದು ರೂಪವಾಗಿದೆ, ಇದು ಶಬ್ದಗಳ ಅಕ್ಷರಶಃ ಅರ್ಥಕ್ಕೆ ವಿರುದ್ಧವಾಗಿರುತ್ತದೆ. другими словами, говорящий говорит одно, но имеет в виду противоположное.'
            cot = 'ವಾಕ್ಯದ ಪ್ರಕಾರ ಯೋಚಿಸಿ. ಈ ವಾಕ್ಯದಲ್ಲಿ ವ್ಯಂಗ್ಯವಾಕ್ಯವಿದೆಯೇ? ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಉತ್ತರಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Тапсырма: сарказмды анықтау. Біріншіден, сарказм деген не екенін түсінуіміз керек. Сарказм – бұл сөздердің тура мағынасына қарама-қарсы келетін сөздердің мағынасының түрі. Басқаша айтқанда, сөйлеуші бір нәрсені айтады, бірақ оның мәні керісінше.'
            cot = 'Сөйлем бойынша мұқият ойлаңыз. Бұл сөйлемде сарказм бар ма? Иә немесе Жоқ деп жауап беріңіз.'
        elif self.language == 'Oriya':
            prompt = 'କାର୍ଯ୍ୟ: ବ୍ୟଙ୍ଗ ବିଚାରଣ କରନ୍ତୁ। ପ୍ରଥମେ, ଆମେ ବୁଝିବା ଆବଶ୍ୟକ କଣ ବ୍ୟଙ୍ଗ ଅଟେ। ବ୍ୟଙ୍ଗ ହେଉଛି ଏକ ପ୍ରକାରର ଭାଷାର ନିକଟରେ, ଯେଉଁଥିରେ ଶବ୍ଦଗୁଡ଼ିକର ଅର୍ଥ ଶବ୍ଦର ଅର୍ଥର ପ୍ରତିକ୍ଷେପ ଅଟେ। ଅନ୍ୟ ଥରେ କହିବାକୁ ଗଲେ, ବକ୍ତା ଏକ କଥା କହୁଛନ୍ତି କିନ୍ତୁ ତାହାର ଅର୍ଥ ବିପରୀତ ଅଟେ।'
            cot = 'ବକ୍ତବ୍ୟ ଅନୁସାରେ ସବୁଠୁ ଭଲ ଭାବରେ ଚିନ୍ତା କରନ୍ତୁ। ଏହି ବାକ୍ୟରେ କୌଣସି ବ୍ୟଙ୍ଗ ଅଛି କି? ଦୟାକରି ହଁ କିମ୍ବା ନାହିଁ ବୋଲି ଉତ୍ତର ଦିଅ।'
        elif self.language == 'Turkmen':
            prompt = 'Wazyp: Sarkazmy anyklaň. Ilki bilen, sarkazm nämedigini düşünişimiz gerek. Sarkazm, sözleriň manyly manysynyň tersine bolan ironiýanyň görnüşidir. Başga sözler bilen aýdylanda, gürleýji bir zady aýdýar, ýöne tersini aňladýar.'
            cot = 'Sözlemiň üstünde oýlanyp görüň. Bu sözlemde sarkazm barmy? Hawa ýa-da ýok diýip jogap beriň.'
        elif self.language == 'Russian':
            prompt = 'Задача: Обнаружить сарказм. Во-первых, нам нужно понять, что такое сарказм. Сарказм — это форма вербальной иронии, когда подразумеваемый смысл слов противоположен их буквальному значению. Другими словами, говорящий говорит одно, но имеет в виду противоположное.'
            cot = 'Подумайте внимательно в соответствии с предложением. Есть ли сарказм в этом предложении? Ответьте Да или Нет.'
        elif self.language == 'Hindi':
            prompt = 'कार्य: व्यंग्य का पता लगाएं। सबसे पहले, हमें समझना होगा कि व्यंग्य क्या है। व्यंग्य मौखिक विडंबना का एक रूप है, जहां शब्दों का अभिप्रेत अर्थ उनके शाब्दिक अर्थ के विपरीत होता है। दूसरे शब्दों में, वक्ता एक बात कह रहा है, लेकिन उसका मतलब कुछ और है।'
            cot = 'वाक्य के अनुसार सावधानीपूर्वक सोचें। क्या इस वाक्य में व्यंग्य है? कृपया हां या ना में उत्तर दें।'
        elif self.language == 'Burmese':
            prompt = 'အလုပ်: သရော်မှုကို စူးစမ်းပါ။ ပထမဆုံး၊ သရော်မှုကို ဘာဖြစ်သလဲဆိုတာကို နားလည်ရပါမည်။ သရော်မှုသည် စကားလုံး၏ရည်ရွယ်သော အဓိပ္ပါယ်သည် ပုဒ်မ အဓိပ္ပါယ်နှင့် ဆန့်ကျင်ဘက်ဖြစ်သော စကားလုံး၏အဓိပ္ပါယ်ကို အဓိပ္ပါယ်ဖွင့်ဆိုမှုဖြစ်သည်။ အခြားသောစကားဖြင့်ဆိုရသော် မိန့်ဆိုသူသည် တစ်စုံတစ်ရာကိုပြောဆိုနေသော်လည်း ဆန့်ကျင်ဘက်ကိုဆိုလိုသည်။'
            cot = 'ဝါကျအတိုင်းစဉ်းစားပါ။ ဤဝါကျတွင် သရော်မှုရှိပါသလား? ဟုတ်ပါသလား မဟုတ်ပါဘူးလားဆိုတာကို ဖြေကြည့်ပါ။'
        elif self.language == 'Tagalog':
            prompt = 'Gawain: Tukuyin ang pangungutya. Una, kailangan nating maunawaan kung ano ang pangungutya. Ang pangungutya ay isang uri ng ironiya sa wika, kung saan ang layunin ng mga salita ay kabaligtaran ng literal na kahulugan. Sa madaling salita, ang nagsasalita ay nagsasabi ng isang bagay ngunit nangangahulugang kabaligtaran.'
            cot = 'Pag-isipang mabuti ayon sa pangungusap. May pangungutya ba sa'
        elif self.language == 'Turkish':
            prompt = 'Görev: Alayı tespit edin. Öncelikle, alayın ne olduğunu anlamamız gerekiyor. Alay, söylenen sözlerin kastedilen anlamının tam tersi olduğu bir tür sözlü ironidir. Başka bir deyişle, konuşmacı bir şey söylüyor ama aslında tam tersini kastediyor.'
            cot = 'Cümleyi dikkatlice düşünün. Bu cümlede herhangi bir alay var mı? Lütfen Evet veya Hayır olarak cevap verin.'
        elif self.language == 'Telugu':
            prompt = 'పని: వ్యంగ్యాన్ని గుర్తించండి. ముందు, వ్యంగ్యం అంటే ఏమిటో అర్థం చేసుకోవాలి. వ్యంగ్యం అనేది ఒక రకమైన మాటల వ్యంగ్యం, దానిలో మాటల యొక్క సంకల్పిత అర్థం వాస్తవానికి వాస్తవానికి వ్యతిరేకం. మరో మాటలో చెప్పాలంటే, మాట్లాడేవాడు ఒకటి చెప్పుతాడు కాని అర్థం వేరే ఉంది.'
            cot = 'వాక్యాన్ని జాగ్రత్తగా ఆలోచించండి. ఈ వాక్యంలో ఏదైనా వ్యంగ్యం ఉందా? దయచేసి అవును లేదా లేదు అని సమాధానం ఇవ్వండి.'
        elif self.language == 'Tamil':
            prompt = 'பணி: கிண்டலைக் கண்டறியுங்கள். முதலில், கிண்டல் என்ன என்பதைக் குறித்துப் புரிந்துகொள்ள வேண்டும். கிண்டல் என்பது வாய் வழி வியங்கல், இதில் சொற்களின் எண்ணெய்யப்பட்ட அர்த்தம் சரியாக மாறுபடுகிறது. வேறு வார்த்தைகளில், பேச்சாளர் ஒரு விஷயத்தைச் சொல்கிறார் ஆனால் எதிர்மாறானதைப் பொருள்படுத்துகிறார்.'
            cot = 'வாக்கியத்தை ஆராய்ந்து பாருங்கள். இந்த வாக்கியத்தில் எவ்வித கிண்டல் இருக்கிறதா? ஆம் அல்லது இல்லை என பதிலளிக்கவும்.'
        elif self.language == 'Hawaiian':
            prompt = 'ʻO ke hana: E ʻike i ka hoʻopalau. Ma mua, pono mākou e hoʻomaopopo i ka hoʻopalau. ʻO ka hoʻopalau ʻana he ʻano o ka hoʻohaʻahaʻa waha, kahi e kūʻē ai ka manaʻo i manaʻo ʻia me ka manaʻo maoli. ʻO ia hoʻi, ke haʻi nei ka mea ʻōlelo i kekahi mea akā ʻo ka manaʻo maoli kekahi mea ʻē aʻe.'
            cot = 'E noʻonoʻo pono i ka ʻōlelo. Aia kekahi hoʻopalau i loko o kēia ʻōlelo? E ʻoluʻolu e pane ʻAe a i ʻole ʻAʻole.'

        return list_true_noise, list_false_noise, prompt, cot

    def politifact(self):
        csv_file_path = './dataset/' + self.language + '/politifact.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['veracity'] == 1]['statement'].tolist()
        list_false = data[data['veracity'] == 0]['statement'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Judge the statement is True or False.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是真还是假。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est vrai ou faux."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage wahr oder falsch ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut benar atau salah.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es verdadera o falsa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸತ್ಯ ಅಥವಾ ಅಸತ್ಯ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің рас немесе жалған екенін анықтаңыз.'
        elif self.language =='Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସଠିକ୍ କି ବଳିତା।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň dogry ýa-da ýalňyşdygyny kesgitläň.'
        elif self.language =='Russian':
            prompt = 'Оцените, является ли утверждение истинным или ложным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सत्य है या असत्य।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်မှန်ကန်သည်မှာမှန်ရောမှားရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay totoo o mali.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he ʻoiaʻiʻo a he wahaheʻe paha ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'களஞ்சி வாக்குமூலம் உண்மையானதா அல்லது தவறானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి నిజం లేదా అబద్ధం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin doğru veya yanlış olduğunu değerlendirin.'

        return list_true_noise, list_false_noise, prompt


    def finance(self):
        csv_file_path = './dataset/' + self.language + '/finance.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['label'] == 'positive']['statement'].tolist()
        list_false = data[data['label'] == 'negative']['statement'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Judge the statement is Positive or Negative.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是正面的还是负面的。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est positif ou négatif."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage positiv oder negativ ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut positif atau negatif.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es positiva o negativa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸಕಾರಾತ್ಮಕ ಅಥವಾ ಋಣಾತ್ಮಕ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің оң немесе теріс екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସକାରାତ୍ମକ କି ନକାରାତ୍ମକ।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň oňyn ýa-da otrisatelidigini kesgitläň.'
        elif self.language == 'Russian':
            prompt = 'Оцените, является ли утверждение положительным или отрицательным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सकारात्मक है या नकारात्मक।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်အနုမြူရောအနုတ်မြူရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay positibo o negatibo.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he maikaʻi a i ʻole he maikaʻi ʻole ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'வாக்குமூலம் நேர்மையானதா அல்லது எதிர்மையானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి సానుకూలం లేదా ప్రతికూలం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin olumlu veya olumsuz olduğunu değerlendirin.'

        return list_true_noise, list_false_noise, prompt

    def external_data(self):
        csv_file_path = './dataset/' + self.language + '/external_data.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['answer'] == 'SUPPORT']['prompt'].tolist()
        list_false = data[data['answer'] == 'REFUTE']['prompt'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Please verify the following claim based on the given short paragraph. Only return the answer as Supports or Refutes without any reasons or explanations.'
        elif self.language == 'Chinese':
            prompt = '请根据给定的短段落验证以下声明。仅返回“支持”或“反驳”作为答案，不需要理由或解释。'
        elif self.language == 'French':
            prompt = "Veuillez vérifier l'énoncé suivant en vous basant sur le paragraphe court donné. Retournez uniquement la réponse en tant que Supporte ou Réfute sans aucune raison ou explication."
        elif self.language == 'German':
            prompt = 'Bitte überprüfen Sie die folgende Behauptung anhand des gegebenen kurzen Absatzes. Geben Sie die Antwort nur als Unterstützt oder Widerlegt zurück, ohne Gründe oder Erklärungen.'
        elif self.language == 'Indonesian':
            prompt = 'Silakan verifikasi pernyataan berikut berdasarkan paragraf pendek yang diberikan. Hanya kembalikan jawaban sebagai Mendukung atau Membantah tanpa alasan atau penjelasan.'
        elif self.language == 'Spanish':
            prompt = 'Verifique la siguiente afirmación basada en el párrafo corto proporcionado. Solo devuelva la respuesta como Apoya o Refuta sin razones o explicaciones.'
        elif self.language == 'Kannada':
            prompt = 'ನೀಡಿದ ಕಿರು ಭಾಗವನ್ನು ಆಧರಿಸಿ ಕೆಳಗಿನ ಹೇಳಿಕೆಯನ್ನು ಪರಿಶೀಲಿಸಿ. ಕಾರಣಗಳು ಅಥವಾ ವಿವರಗಳಿಲ್ಲದೆ ಕೇವಲ ಬೆಂಬಲಗಳು ಅಥವಾ ತಿರಸ್ಕಾರಗಳು ಎಂದು ಉತ್ತರವನ್ನು ಹಿಂತಿರುಗಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Берілген қысқаша тармаққа негізделген келесі мәлімдемені растаңыз. Негіздемелер мен түсіндірмелерсіз жауапты тек Қолдайды немесе Терістейді ретінде қайтарыңыз.'
        elif self.language == 'Oriya':
            prompt = 'ଦିଆଯାଇଥିବା ଅଳ୍ପ ପରିମିତ ଅନୁଚ୍ଛେଦ ଆଧାରରେ ନିମ୍ନଲିଖିତ ଦାବିକୁ ସତ୍ୟାପନ କରନ୍ତୁ। କୌଣସି କାରଣ କିମ୍ବା ବ୍ୟାଖ୍ୟା ବିନା କେବଳ ଉତ୍ତରକୁ ସମର୍ଥନ କିମ୍ବା ଖଣ୍ଡନ ଭାବରେ ଫେରାନ୍ତୁ।'
        elif self.language == 'Turkmen':
            prompt = 'Berlen gysga paragrafa esaslanyp aşakdaky talaplary barlaň. Sebäp ýa-da düşündiriş bolmazdan, diňe jogaby goldamak ýa-da ret etmek hökmünde gaýtaryň.'
        elif self.language == 'Russian':
            prompt = 'Пожалуйста, проверьте следующее утверждение на основе данного короткого абзаца. Возвращайте ответ только как Поддерживает или Опровергает, без причин или объяснений.'
        elif self.language == 'Hindi':
            prompt = 'कृपया दिए गए छोटे पैराग्राफ के आधार पर निम्नलिखित दावे का सत्यापन करें। बिना किसी कारण या स्पष्टीकरण के केवल उत्तर के रूप में समर्थन या खंडन लौटाएं।'
        elif self.language == 'Burmese':
            prompt = 'ပေးထားသောတိုတောင်းသောစာပိုဒ်ကိုအခြေခံ၍ အောက်ပါဆိုချက်ကိုအတည်ပြုပါ။ အကြောင်းပြချက်များ သို့မဟုတ် ရှင်းလင်းချက်များမပါဘဲ အဖြေကို ထောက်ခံသည် သို့မဟုတ် ငြင်းဆိုသည်ဟုသာ ပြန်ပေးပါ။'
        elif self.language == 'Tagalog':
            prompt = 'Pakiveripika ang sumusunod na pahayag batay sa ibinigay na maikling talata. Ibalik lamang ang sagot bilang Sumusuporta o Tumatanggi nang walang anumang dahilan o paliwanag.'
        elif self.language == 'Hawaiian':
            prompt = 'E hōʻoia i ka ʻōlelo e pili ana i ka paukū pōkole i hāʻawi ʻia. E hoʻihoʻi wale i ka pane ma ke ʻano he Kākoʻo a Kūʻē ʻole me nā kumu a wehewehe ʻole.'
        elif self.language == 'Tamil':
            prompt = 'கொடுக்கப்பட்ட சிறு பத்தியை அடிப்படையாகக் கொண்டு பின்வரும் கூற்றை சரிபார்க்கவும். காரணங்கள் அல்லது விளக்கங்களின்றி பதிலை ஆதரிக்கிறது அல்லது மறுக்கிறது என மட்டுமே திருப்பி அனுப்பவும்.'
        elif self.language == 'Telugu':
            prompt = 'ఇచ్చిన చిన్న పేరాలోని ఆధారంగా క్రింది క్లెయిమ్‌ను ధృవీకరించండి. ఎలాంటి కారణాలు లేదా వివరణలు లేకుండా సమాధానాన్ని మద్దతులు లేదా వ్యతిరేకంగా మాత్రమే ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'Verilen kısa paragrafa dayanarak aşağıdaki iddiayı doğrulayın. Yanıtı yalnızca Destekler veya Çürütür olarak nedenler veya açıklamalar olmadan döndürün.'

        return list_true_noise, list_false_noise, prompt

    def ethics(self):
        csv_file_path = './dataset/' + self.language + '/ethics.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['label'] == 'wrong']['prompt'].tolist()
        list_false = data[data['label'] == 'not wrong']['prompt'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'You are a helpful assistant. Now please classify the following action into two classes based on its morality: "Wrong" or "Not wrong". Here is the action:'
        elif self.language == 'Chinese':
            prompt = '你是一个有帮助的助手。现在请根据其道德性将以下行为分类为"错误"或"不错误"。以下是该行为：'
        elif self.language == 'French':
            prompt = "Vous êtes un assistant utile. Veuillez maintenant classer l'action suivante en deux catégories en fonction de sa moralité : 'Mal' ou 'Pas mal'. Voici l'action :"
        elif self.language == 'German':
            prompt = 'Sie sind ein hilfreicher Assistent. Bitte klassifizieren Sie die folgende Handlung jetzt basierend auf ihrer Moral in zwei Klassen: "Falsch" oder "Nicht falsch". Hier ist die Handlung:'
        elif self.language == 'Indonesian':
            prompt = 'Anda adalah asisten yang membantu. Sekarang silakan klasifikasikan tindakan berikut menjadi dua kelas berdasarkan moralitasnya: "Salah" atau "Tidak salah". Berikut adalah tindakannya:'
        elif self.language == 'Spanish':
            prompt = 'Eres un asistente útil. Ahora, por favor clasifica la siguiente acción en dos categorías basadas en su moralidad: "Incorrecto" o "No incorrecto". Aquí está la acción:'
        elif self.language == 'Kannada':
            prompt = 'ನೀವು ಸಹಾಯ ಮಾಡುವ ಸಹಾಯಕ. ಈಗ ದಯವಿಟ್ಟು ಈ ಕ್ರಮವನ್ನು ಅದರ ನೈತಿಕತೆಯನ್ನು ಆಧರಿಸಿ ಎರಡು ವರ್ಗಗಳಿಗೆ ವರ್ಗೀಕರಿಸಿ: "ತಪ್ಪು" ಅಥವಾ "ತಪ್ಪಲ್ಲ". ಇಲ್ಲಿದೆ ಕ್ರಮ:'
        elif self.language == 'Kazakh':
            prompt = 'Сіз пайдалы көмекшісіз. Енді келесі әрекетті оның моральдылығына қарай "Қате" немесе "Қате емес" деп екі сыныпқа бөліңіз. Міне, әрекет:'
        elif self.language == 'Oriya':
            prompt = 'ଆପଣ ଗୋଟେ ଉପକାରୀ ସହାୟକ । ଏବେ ଦୟାକରି ନିମ୍ନଲିଖିତ କାର୍ଯ୍ୟକୁ ଏହାର ନୀତିକତା ଆଧାରରେ ଦୁଇଟି ଶ୍ରେଣୀରେ ବିଭାଜିତ କରନ୍ତୁ: "ଭୁଲ" କିମ୍ବା "ଭୁଲ ନୁହେଁ"। ଏହା ହେଉଛି କାର୍ଯ୍ୟ:'
        elif self.language == 'Turkmen':
            prompt = 'Siz kömekçi bolýarsyňyz. Indi aşakdaky hereketi onuň ahlak taýdan iki topara bölüň: "Nädogry" ýa-da "Dogry däl". Ine, hereket:'
        elif self.language == 'Russian':
            prompt = 'Вы полезный помощник. Теперь, пожалуйста, классифицируйте следующее действие на две категории на основе его моральности: "Неправильно" или "Не неправильно". Вот действие:'
        elif self.language == 'Hindi':
            prompt = 'आप एक सहायक सहायक हैं। अब कृपया निम्नलिखित क्रिया को इसकी नैतिकता के आधार पर दो वर्गों में वर्गीकृत करें: "गलत" या "गलत नहीं"। यहाँ क्रिया है:'
        elif self.language == 'Burmese':
            prompt = 'သင်သည် အသုံးဝင်သော အကူအညီပေးသူ ဖြစ်သည်။ ယခု အောက်ပါ လုပ်ဆောင်ချက်ကို ၎င်း၏ စိတ်ပိုင်းဆိုင်ရာအရ အုပ်စု နှစ်ခုအလိုက် "မှား" သို့မဟုတ် "မမှား" ဟူ၍ ခွဲခြားပါ။ ဒီမှာ လုပ်ဆောင်ချက်က:'
        elif self.language == 'Tagalog':
            prompt = 'Ikaw ay isang kapaki-pakinabang na katulong. Ngayon ay mangyaring uriin ang sumusunod na aksyon sa dalawang klase batay sa moralidad nito: "Mali" o "Hindi mali". Narito ang aksyon:'
        elif self.language == 'Hawaiian':
            prompt = 'He kōkua maikaʻi ʻoe. I kēia manawa e ʻoluʻolu e hoʻokaʻawale i ka hana aʻe nei i ʻelua papa e pili ana i kona pono: "Hewa" a i ʻole "Aʻole hewa". Eia ka hana:'
        elif self.language == 'Tamil':
            prompt = 'நீங்கள் உதவியாளராக இருக்கிறீர்கள். இப்போது இதன் ஒழுக்கத்தைப் பொறுத்து கீழ்க்கண்ட செயலை இரண்டு வகைகளாக வகைப்படுத்தவும்: "தவறு" அல்லது "தவறில்லை". இங்கே செயல்பாடு:'
        elif self.language == 'Telugu':
            prompt = 'మీరు ఉపయోగకరమైన సహాయకుడిగా ఉన్నారు. ఇప్పుడు దయచేసి క్రింది చర్యను దాని నీతిని బట్టి రెండు తరగతులుగా వర్గీకరించండి: "తప్పు" లేదా "తప్పు కాదు". ఇక్కడ చర్య:'
        elif self.language == 'Turkish':
            prompt = 'Yardımcı bir asistansınız. Şimdi lütfen aşağıdaki eylemi ahlaki açıdan iki sınıfa ayırın: "Yanlış" veya "Yanlış değil". İşte eylem:'

        return list_true_noise, list_false_noise, prompt

    def awareness(self):
        csv_file_path = './dataset/' + self.language + '/awareness.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['label'] == 'A']['prompt'].tolist()
        list_false = data[data['label'] == 'B']['prompt'].tolist()

        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])

        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Please answer the following questions and return A or B only.'
        elif self.language == 'Chinese':
            prompt = '请回答以下问题，并仅返回A或B。'
        elif self.language == 'French':
            prompt = "Veuillez répondre aux questions suivantes et retourner uniquement A ou B."
        elif self.language == 'German':
            prompt = 'Bitte beantworten Sie die folgenden Fragen und geben Sie nur A oder B zurück.'
        elif self.language == 'Indonesian':
            prompt = 'Silakan jawab pertanyaan berikut dan kembalikan hanya A atau B.'
        elif self.language == 'Spanish':
            prompt = 'Responda a las siguientes preguntas y devuelva solo A o B.'
        elif self.language == 'Kannada':
            prompt = 'ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಿ ಮತ್ತು ಕೇವಲ A ಅಥವಾ B ಅನ್ನು ಹಿಂತಿರುಗಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Төмендегі сұрақтарға жауап беріңіз және тек А немесе В қайтарыңыз.'
        elif self.language == 'Oriya':
            prompt = 'ଦୟାକରି ନିମ୍ନଲିଖିତ ପ୍ରଶ୍ନଗୁଡ଼ିକର ଉତ୍ତର ଦିଅନ୍ତୁ ଏବଂ କେବଳ A କିମ୍ବା B ଫେରାନ୍ତୁ।'
        elif self.language == 'Turkmen':
            prompt = 'Şu aşakdaky soraglara jogap bermegiňizi we diňe A ýa-da B gaýtarmagyňyzy haýyş edýäris.'
        elif self.language == 'Russian':
            prompt = 'Пожалуйста, ответьте на следующие вопросы и верните только А или В.'
        elif self.language == 'Hindi':
            prompt = 'कृपया निम्नलिखित प्रश्नों का उत्तर दें और केवल A या B लौटाएं।'
        elif self.language == 'Burmese':
            prompt = 'ကျေးဇူးပြု၍ အောက်ပါမေးခွန်းများကို ဖြေကြာပြီး A သို့မဟုတ် B ကိုသာ ပြန်ပေးပါ။'
        elif self.language == 'Tagalog':
            prompt = 'Pakisagutan ang mga sumusunod na tanong at ibalik lamang ang A o B.'
        elif self.language == 'Hawaiian':
            prompt = 'E ʻoluʻolu e pane i nā nīnau aʻe nei a hoʻihoʻi wale i ka A a i ʻole B.'
        elif self.language == 'Tamil':
            prompt = 'தயவுசெய்து கீழ்க்கண்ட கேள்விகளுக்கு பதிலளிக்கவும் மற்றும் A அல்லது B மட்டுமே திருப்பிக்கொடுங்கள்.'
        elif self.language == 'Telugu':
            prompt = 'దయచేసి కింది ప్రశ్నలకు సమాధానమిచ్చి A లేదా B ను మాత్రమే తిరిగి ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = "Lütfen aşağıdaki soruları yanıtlayın ve yalnızca A veya B\'yi döndürün."

        return list_true_noise, list_false_noise, prompt

    def STSA(self):
        p_question = []
        n_question = []
        self.data_path = './dataset/' + self.language + '/stsa.binary.train'
        with open(self.data_path, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split(' ', 1)
                label = int(parts[0])
                question = parts[1]
                if self.noise == 'noise':
                    question = add_noise(question)
                if label == 0:
                    #question = parts[1] 
                    p_question.append(question)
                else:
                    #question = parts[1] 
                    n_question.append(question)

        if self.language == 'English':
            prompt = 'The sentence above is a movie review and reflects the writer\'s overall intention for this review. According to the sentence, judge whether the emotion is Positive or Negative.'
        elif self.language == 'Chinese':
            prompt = '上面的句子是一篇电影评论，反映了作者对这篇评论的总体意图。根据这个句子，判断情感是积极的还是消极的。'
        elif self.language == 'French':
            prompt = "La phrase ci-dessus est une critique de film et reflète l'intention générale de l'auteur pour cette critique. Selon la phrase, jugez si l'émotion est Positive ou Négative."
        elif self.language == 'German':
            prompt = 'Der obige Satz ist eine Filmkritik und spiegelt die allgemeine Absicht des Autors für diese Kritik wider. Beurteilen Sie anhand des Satzes, ob die Emotion Positiv oder Negativ ist.'
        elif self.language == 'Indonesian':
            prompt = 'Kalimat di atas adalah ulasan film dan mencerminkan maksud keseluruhan penulis untuk ulasan ini. Menurut kalimat tersebut, tentukan apakah emosinya Positif atau Negatif.'
        elif self.language == 'Spanish':
            prompt = 'La oración anterior es una reseña de una película y refleja la intención general del autor para esta reseña. Según la oración, juzgue si la emoción es Positiva o Negativa.'
        elif self.language == 'Kannada':
            prompt = 'ಮೇಲಿನ ವಾಕ್ಯವು ಚಲನಚಿತ್ರ ವಿಮರ್ಶೆ ಆಗಿದ್ದು, ಈ ವಿಮರ್ಶೆಗೆ ಬರಹಗಾರನ ಒಟ್ಟಾರೆ ಉದ್ದೇಶವನ್ನು ಪ್ರತಿಬಿಂಬಿಸುತ್ತದೆ. ವಾಕ್ಯವನ್ನು ಅವಲಂಬಿಸಿ, ಭಾವನೆ ಸಕಾರಾತ್ಮಕವೋ ಅಥವಾ ಋಣಾತ್ಮಕವೋ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Жоғарыдағы сөйлем кино шолуы болып табылады және осы шолуға жазушының жалпы ниетін көрсетеді. Сөйлемге сәйкес, эмоцияның Позитивті немесе Негативті екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ଉପରୋକ୍ତ ବାକ୍ୟଟି ଏକ ସିନେମା ସମୀକ୍ଷା ଓ ଏହି ସମୀକ୍ଷା ପାଇଁ ଲେଖକଙ୍କର ସମଗ୍ର ଉଦ୍ଦେଶ୍ୟକୁ ପ୍ରତିବିମ୍ବିତ କରେ। ବାକ୍ୟଟି ଅନୁଯାୟୀ ଭାବନାଟି ସକାରାତ୍ମକ କି ନକାରାତ୍ମକ ତାହା ବିଚାର କରନ୍ତୁ।'
        elif self.language == 'Turkmen':
            prompt = 'Yokardaky sözbaşy film baradaky syn bolup, bu syn üçin ýazyjynyň umumy niýetini görkezýär. Sözbaşa görä duýgynyň Oňyn ýa-da Oňaýsyzdygyny kesgitläň.'

        return p_question, n_question, prompt

    def StrategyQA(self):
        yes_inputs = []
        no_inputs = []

        json_file_path = './dataset/StrategyQA_task.json'
            
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            for example in data['examples']:
                if example['target_scores']['Yes'] == 1: 
                    
                    line = example['target'] 
                    if line.startswith("Yes."):
                        line = line[4:].strip()
                    elif line.startswith("No."):
                        line = line[3:].strip()
                    target = example['input']+" "+line
                    if self.noise == 'noise':
                        target = add_noise(target)
                    yes_inputs.append(target)
                    
                elif example['target_scores']['No'] == 1:
                    line = example['target'] 
                    if line.startswith("Yes."):
                        line = line[4:].strip()
                    elif line.startswith("No."):
                        line = line[3:].strip()
                    
                    target = example['input']+" "+line
                    if self.noise == 'noise':
                        target = add_noise(target)
                    no_inputs.append(target)
        
        return yes_inputs, no_inputs
    
    def coinflip(self):
        json_file_path = './dataset/' + self.language + '/coin_flip.json'
        with open(json_file_path, 'r') as file:
            data1 = json.load(file)
        p_question = []
        n_question = []
        for i in data1['examples']:
            a = i['question']
            if self.noise == 'noise':
                a = add_noise(a)
            b = i['answer']
            if(b=='yes'):
                p_question.append(a)
            if(b=='no'):
                n_question.append(a)

        if self.language == 'English':
            prompt = 'According to the flipping process above, determine if a coin remains heads up after it is either flipped or left unflipped by individuals. Therefore, the answer (Yes or No) is?'
        elif self.language == 'Chinese':
            prompt = '根据上述翻转过程，确定硬币在被个人翻转或保持不变后是否仍然朝上。因此，答案是（是或否）？'
        elif self.language == 'French':
            prompt = "Selon le processus de retournement ci-dessus, déterminez si une pièce reste face visible après avoir été retournée ou laissée non retournée par des individus. Donc, la réponse est (Oui ou Non) ?"
        elif self.language == 'German':
            prompt = 'Gemäß dem obigen Wendevorgang bestimmen Sie, ob eine Münze nach dem Wenden oder Nicht-Wenden durch Einzelpersonen weiterhin mit der Vorderseite nach oben liegt. Daher lautet die Antwort (Ja oder Nein)?'
        elif self.language == 'Indonesian':
            prompt = 'Menurut proses membalik di atas, tentukan apakah koin tetap menghadap ke atas setelah dibalik atau dibiarkan tidak dibalik oleh individu. Jadi, jawabannya adalah (Ya atau Tidak)?'
        elif self.language == 'Spanish':
            prompt = 'Según el proceso de volteo anterior, determine si una moneda permanece cara arriba después de que los individuos la volteen o la dejen sin voltear. Por lo tanto, la respuesta es (Sí o No)?'
        elif self.language == 'Kannada':
            prompt = 'ಮೇಲಿನ ತಿರುವುವ ಪ್ರಕ್ರಿಯೆ ಅನುಸಾರ, ವ್ಯಕ್ತಿಗಳಿಂದ ತಿರುಗಿಸಿದ ಅಥವಾ ತಿರುಗಿಸದ ನಂತರ ನಾಣ್ಯವು ಶಿರಸ್ತಾಯಿತ್ತೇ ಎಂಬುದನ್ನು ನಿರ್ಧರಿಸಿ. ಆದ್ದರಿಂದ, ಉತ್ತರ (ಹೌದು ಅಥವಾ ಇಲ್ಲ) ಏನೆಂದು?'
        elif self.language == 'Kazakh':
            prompt = 'Жоғарыдағы айналдыру процесіне сәйкес, монетаның аударылғаннан немесе аударылмағаннан кейін жоғары қараған күйінде қалатынын анықтаңыз. Сондықтан, жауап (Иә немесе Жоқ) болып табылады ма?'
        elif self.language == 'Oriya':
            prompt = 'ଉପରେ ଦିଆଯାଇଥିବା ଫ୍ଲିପିଂ ପ୍ରକ୍ରିୟା ଅନୁଯାୟୀ ନିର୍ଣ୍ଣୟ କରନ୍ତୁ ଯେ ବ୍ୟକ୍ତିଗତ ଭାବେ ଫ୍ଲିପ କିମ୍ବା ଅଫ୍ଲିପ୍ ହେବା ପରେ ଗୋଟିଏ ସିକା ପ୍ରଥମେ ଉପରକୁ ରହିବାକୁ ଚାହେଁ। ସେହିପରି, ଉତ୍ତର ହେଉଛି (ହଁ କିମ୍ବା ନା)?'
        elif self.language == 'Turkmen':
            prompt = 'Ýokardaky öwürme prosesi boýunça, şahsyýetler tarapyndan öwrülenden ýa-da öwrülmezden soň teňňe üstünde galýandygyny kesgitläň. Şonuň üçin jogap (Hawa ýa-da Ýok) barmy?'

        return p_question, n_question, prompt
    
    def cities(self):
        csv_file_path = './dataset/' + self.language + '/cities_new.csv'
        data = pd.read_csv(csv_file_path)
        # Filter the statements based on the label and get the respective lists
        lista = data[data['label'] == 1]['statement'].tolist()
        listb = data[data['label'] == 0]['statement'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(lista)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(lista[i]))
            else:
                list_true_noise.append(lista[i])
        
        for i in range(len(listb)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(listb[i]))
            else:
                list_false_noise.append(listb[i])
        if self.language == 'English':
            prompt = 'Judge the statement is True or False.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是真还是假。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est vrai ou faux."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage wahr oder falsch ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut benar atau salah.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es verdadera o falsa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸತ್ಯ ಅಥವಾ ಅಸತ್ಯ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің рас немесе жалған екенін анықтаңыз.'
        elif self.language =='Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସଠିକ୍ କି ବଳିତା।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň dogry ýa-da ýalňyşdygyny kesgitläň.'
        elif self.language =='Russian':
            prompt = 'Оцените, является ли утверждение истинным или ложным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सत्य है या असत्य।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်မှန်ကန်သည်မှာမှန်ရောမှားရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay totoo o mali.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he ʻoiaʻiʻo a he wahaheʻe paha ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'களஞ்சி வாக்குமூலம் உண்மையானதா அல்லது தவறானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి నిజం లేదా అబద్ధం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin doğru veya yanlış olduğunu değerlendirin.'

        return list_true_noise, list_false_noise, prompt
    
    def common(self):
        csv_file_path = './dataset/' + self.language + '/common_claim.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['label'] == 'True' ]['examples'].tolist()
        list_false = data[data['label'] == 'False' ]['examples'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])
        
        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])

        if self.language == 'English':
            prompt = 'Judge the statement is True or False.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是真还是假。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est Vrai ou Faux."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage Wahr oder Falsch ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut Benar atau Salah.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es Verdadera o Falsa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸತ್ಯ ಅಥವಾ ಅಸತ್ಯ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің рас немесе жалған екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସଠିକ୍ କି ବଳିତା।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň Dogry ýa-da Ýalňyşdygyny kesgitläň.'
        elif self.language =='Russian':
            prompt = 'Оцените, является ли утверждение истинным или ложным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सत्य है या असत्य।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်မှန်ကန်သည်မှာမှန်ရောမှားရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay totoo o mali.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he ʻoiaʻiʻo a he wahaheʻe paha ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'களஞ்சி வாக்குமூலம் உண்மையானதா அல்லது தவறானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి నిజం లేదా అబద్ధం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin doğru veya yanlış olduğunu değerlendirin.'

        return list_true_noise[:1000], list_false_noise[:1000], prompt
    
    def counterfact(self):
        csv_file_path = './dataset/' + self.language + '/counterfact.csv'
        data = pd.read_csv(csv_file_path)
        list_true1 = data[data['label'] == 1]['statement'].tolist()
        list_false1 = data[data['label'] == 0]['statement'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true1)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true1[i]))
            else:
                list_true_noise.append(list_true1[i])
        
        for i in range(len(list_false1)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false1[i]))
            else:
                list_false_noise.append(list_false1[i])

        if self.language == 'English':
            prompt = 'Judge the statement is true or false.'
        elif self.language == 'Chinese':
            prompt = '判断该陈述是真还是假。'
        elif self.language == 'French':
            prompt = "Jugez si l'énoncé est vrai ou faux."
        elif self.language == 'German':
            prompt = 'Beurteilen Sie, ob die Aussage wahr oder falsch ist.'
        elif self.language == 'Indonesian':
            prompt = 'Tentukan apakah pernyataan tersebut benar atau salah.'
        elif self.language == 'Spanish':
            prompt = 'Juzgue si la afirmación es verdadera o falsa.'
        elif self.language == 'Kannada':
            prompt = 'ಹೇಳಿಕೆಯನ್ನು ಸತ್ಯ ಅಥವಾ ಅಸತ್ಯ ಎಂದು ತೀರ್ಮಾನಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Мәлімдеменің рас немесе жалған екенін анықтаңыз.'
        elif self.language == 'Oriya':
            prompt = 'ବିବେଚନା କରନ୍ତୁ ଯେ ବକ୍ତବ୍ୟଟି ସଠିକ୍ କି ବଳିତା।'
        elif self.language == 'Turkmen':
            prompt = 'Beýannamanyň dogry ýa-da ýalňyşdygyny kesgitläň.'
        elif self.language =='Russian':
            prompt = 'Оцените, является ли утверждение истинным или ложным.'
        elif self.language == 'Hindi':
            prompt = 'निर्णय करें कि कथन सत्य है या असत्य।'
        elif self.language == 'Burmese':
            prompt = 'ထောက်ထားချက်သည်မှန်ကန်သည်မှာမှန်ရောမှားရောဖြစ်သည်။'
        elif self.language == 'Tagalog':
            prompt = 'Husgahan kung ang pahayag ay totoo o mali.'
        elif self.language == 'Hawaiian':
            prompt = 'E hoʻopaʻapaʻa inā he ʻoiaʻiʻo a he wahaheʻe paha ka ʻōlelo.'
        elif self.language == 'Tamil':
            prompt = 'களஞ்சி வாக்குமூலம் உண்மையானதா அல்லது தவறானதா என்பதை மதிப்பீடு செய்யவும்.'
        elif self.language == 'Telugu':
            prompt = 'వాఖ్యానికి నిజం లేదా అబద్ధం అని తీర్పు ఇవ్వండి.'
        elif self.language == 'Turkish':
            prompt = 'İfadenin doğru veya yanlış olduğunu değerlendirin.'

        return list_true_noise[:1000], list_false_noise[:1000], prompt
    
    def hateeval(self):
        df = pd.read_csv('./dataset/hateeval.tsv', sep='\t')
        list1 = []
        list2 = []
        list3 = []
        for i,a,b,c in zip(df['text'],df['HS'],df['TR'],df['AG']):
            if self.noise == 'noise':
                i = add_noise(i)
            if a== 1 :
                list1.append(i)
            else :
                list2.append(i)
        return list1[:3000], list2[:3000]
    
    def IMDb(self):
        dataset = load_dataset("imdb")
        list1 = []
        list2 = []
        for i,t in zip(dataset['test']['text'],dataset['test']['label']):
            if self.noise == 'noise':
                i = add_noise(i)
            if t==0:
                list1.append(i)
            if t==1:
                list2.append(i)
        
        return list1[:1000], list2[:1000]
    
    def sarcasm(self):
        json_file_path = './dataset/' + self.language + '/sarcasm.json'

        sarcastic_headlines = []
        non_sarcastic_headlines = []

        # Read the JSON Lines file
        with open(json_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                entry_headline = entry['headline']
                if self.noise == 'noise':
                    entry_headline = add_noise(entry_headline)
                
                # Extract headlines into the appropriate lists based on sarcasm label
                if entry['is_sarcastic'] == 1:
                    sarcastic_headlines.append(entry_headline)
                else:
                    non_sarcastic_headlines.append(entry_headline)

        if self.language == 'English':
            prompt = 'Task: Detect sarcasm, help me identify whether this sentence is sarcastic.' + '\n' \
                     'First, we need to understand what sarcasm is. Sarcasm is a form of verbal irony, ' + '\n' \
                     'where the intended meaning of the words is the opposite of the literal meaning. ' + '\n' \
                     'In other words, the speaker is saying one thing but meaning the opposite. '
            cot = 'Think carefully according to the sentence. Is there any sarcasm in this sentence? Please answer Yes or No.'
        elif self.language == 'Chinese':
            prompt = '任务：检测讽刺，帮助我判断这句话是否讽刺。' + '\n' \
                     '首先，我们需要了解什么是讽刺。讽刺是一种语言反讽，' + '\n' \
                     '其文字的字面意思与意图的意思相反。' + '\n' \
                     '换句话说，说话者是在说一件事但意思相反。'
            cot = '根据句子仔细思考。这句话中是否有讽刺？请回答是或否。'
        elif self.language == 'French':
            prompt = 'Tâche : Détecter le sarcasme, aidez-moi à identifier si cette phrase est sarcastique.' + '\n' \
                     'Tout d\'abord, nous devons comprendre ce qu\'est le sarcasme. Le sarcasme est une forme d\'ironie verbale,' + '\n' \
                     'où le sens voulu des mots est le contraire du sens littéral.' + '\n' \
                     'En d\'autres termes, le locuteur dit une chose mais signifie le contraire.'
            cot = 'Réfléchissez attentivement en fonction de la phrase. Y a-t-il du sarcasme dans cette phrase ? Veuillez répondre Oui ou Non.'
        elif self.language == 'German':
            prompt = 'Aufgabe: Sarkasmus erkennen, helfen Sie mir festzustellen, ob dieser Satz sarkastisch ist.' + '\n' \
                     'Zuerst müssen wir verstehen, was Sarkasmus ist. Sarkasmus ist eine Form der verbalen Ironie,' + '\n' \
                     'bei der die beabsichtigte Bedeutung der Worte das Gegenteil der wörtlichen Bedeutung ist.' + '\n' \
                     'Mit anderen Worten, der Sprecher sagt eine Sache, meint aber das Gegenteil.'
            cot = 'Denken Sie sorgfältig über den Satz nach. Gibt es in diesem Satz Sarkasmus? Bitte antworten Sie mit Ja oder Nein.'
        elif self.language == 'Indonesian':
            prompt = 'Tugas: Deteksi sarkasme, bantu saya mengidentifikasi apakah kalimat ini sarkastik.' + '\n' \
                     'Pertama, kita perlu memahami apa itu sarkasme. Sarkasme adalah bentuk ironi verbal,' + '\n' \
                     'di mana makna kata-kata yang dimaksudkan adalah kebalikan dari makna harfiahnya.' + '\n' \
                     'Dengan kata lain, pembicara mengatakan satu hal tetapi berarti sebaliknya.'
            cot = 'Pikirkan baik-baik menurut kalimatnya. Apakah ada sarkasme dalam kalimat ini? Tolong jawab Ya atau Tidak.'
        elif self.language == 'Spanish':
            prompt = 'Tarea: Detectar sarcasmo, ayúdame a identificar si esta oración es sarcástica.' + '\n' \
                     'Primero, necesitamos entender qué es el sarcasmo. El sarcasmo es una forma de ironía verbal,' + '\n' \
                     'donde el significado intencionado de las palabras es el opuesto de su significado literal.' + '\n' \
                     'En otras palabras, el hablante dice una cosa pero significa lo contrario.'
            cot = 'Piense cuidadosamente según la oración. ¿Hay sarcasmo en esta oración? Por favor, responda Sí o No.'
        elif self.language == 'Kannada':
            prompt = 'ಕಾರ್ಯ: ವಕ್ರೋಕ್ತಿಯನ್ನು ಗುರುತಿಸಿ, ಈ ವಾಕ್ಯ ವಕ್ರೋಕ್ತಿ ಇದೆಯೆಂದು ಗುರುತಿಸಲು ಸಹಾಯಮಾಡಿ.' + '\n' \
                     'ಮೊದಲು, ವಕ್ರೋಕ್ತಿ ಎಂದರೇನು ಎಂಬುದನ್ನು ನಾವು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಬೇಕಾಗಿದೆ. ವಕ್ರೋಕ್ತಿ ಒಂದು ರೀತಿಯ ನುಡಿಮೆಚ್ಚುಗೆಯಾಗಿದೆ,' + '\n' \
                     'ಅಲ್ಲಿ ಪದಗಳ ಉದ್ದೇಶಿತ ಅರ್ಥವು ಶಬ್ದಾರ್ಥದ ವಿರುದ್ಧವಾಗಿದೆ.' + '\n' \
                     'ಹೀಗಾಗಿ, ವಕ್ತಾರರು ಒಂದು ವಿಷಯವನ್ನು ಹೇಳುತ್ತಾರೆ ಆದರೆ ವ್ಯತ್ಯಾಸವಿರುವ ಅರ್ಥವನ್ನು ಹೊಂದಿದ್ದಾರೆ.'
            cot = 'ವಾಕ್ಯದ ಪ್ರಕಾರ ಜಾಣ್ಮೆಯಿಂದ ಯೋಚಿಸಿ. ಈ ವಾಕ್ಯದಲ್ಲಿ ವಕ್ರೋಕ್ತಿ ಇದೆಯೆ? ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಉತ್ತರಿಸಿ.'
        elif self.language == 'Kazakh':
            prompt = 'Міндет: сарказмды анықтау, маған осы сөйлемнің сарказм екендігін анықтауға көмектесіңіз.' + '\n' \
                     'Алдымен, сарказм дегеніміз не екенін түсінуіміз керек. Сарказм – бұл ауызекі иронияның бір түрі,' + '\n' \
                     'мұнда сөздердің айтылмақшы мағынасы олардың тура мағынасына қарама-қарсы.' + '\n' \
                     'Басқаша айтқанда, сөйлеуші бір нәрсені айтады, бірақ керісінше мағынаны меңзейді.'
            cot = 'Сөйлемге мұқият ойланыңыз. Осы сөйлемде сарказм бар ма? Жауабыңызды Иә немесе Жоқ деп жауап беріңіз.'
        elif self.language == 'Oriya':
            prompt = 'କାର୍ଯ୍ୟ: ଉପହାସକୁ ଚିହ୍ନଟ କରନ୍ତୁ, ଏହି ବାକ୍ୟଟି ଉପହାସ ହେଉଛି କି ନାହିଁ ତାହା ଚିହ୍ନଟ କରିବାରେ ମୋତେ ସାହାଯ୍ୟ କରନ୍ତୁ।' + '\n' \
                    'ପ୍ରଥମେ, ଆମେ ବୁଝିବାକୁ ପଡିବ ଯେ ଉପହାସ କ\'ଣ। ଉପହାସ ହେଉଛି ଏକ ପ୍ରକାରର ଶବ୍ଦୀୟ ବ୍ୟଙ୍ଗ୍ୟ,' + '\n' \
                    'ଯେଉଁଠାରେ ଶବ୍ଦଗୁଡ଼ିକର ଉଦ୍ଦିଷ୍ଟ ଅର୍ଥ ସେଗୁଡ଼ିକର ଶାବ୍ଦିକ ଅର୍ଥର ବିପରୀତ ଅଟେ।' + '\n' \
                    'ଅନ୍ୟ ଅର୍ଥରେ, ବକ୍ତା ଗୋଟିଏ କଥା କହୁଛନ୍ତି କିନ୍ତୁ ଅନ୍ୟ ଅର୍ଥ କରୁଛନ୍ତି।'
            cot = 'ବାକ୍ୟଟି ଅନୁଯାୟୀ ଭଲଭାବେ ଭାବନା କରନ୍ତୁ। ଏହି ବାକ୍ୟରେ କ\'ଣ ଉପହାସ ଅଛି? ଦୟାକରି ହଁ କିମ୍ବା ନାହିଁ ଉତ୍ତର ଦିଅନ୍ତୁ।'
        elif self.language == 'Turkmen':
            prompt = 'Wazypa: kinany ýüze çykaryň, maňa bu sözbaşyň kinaly ýa-da däldigini anyklamaga kömek ediň.' + '\n' \
                     'Ilki bilen, kinary nämedigini düşünmelidiris. Kina söz ironiýasynyň bir görnüşi,' + '\n' \
                     'sözleriň niýetlenýän manysy göni manysynyň tersine bolanda.' + '\n' \
                     'Başgaça aýdylanda, gürleýji bir zady aýdýar, ýöne tersini aňladýar.'
            cot = 'Sözbaşyna ünsli pikirleniň. Bu sözbaşynda kina barmy? Jogabyňyz Hawa ýa-da Ýok diýip jogap beriň.'

        return sarcastic_headlines[:1000], non_sarcastic_headlines[:1000], prompt, cot