# 🏡 Airbnb Price Estimate
Bu proje, New York City'deki Airbnb ilanlarının gecelik fiyatlarını tahmin etmeyi amaçlamaktadır. Global AI Hub bootcamp sürecinde regresyon tabanlı tahmin modelleri ile gerçekleştirilmiştir. Bu README, projenin yapısını ve elde edilen bulguları detaylıca sunmaktadır.

# 📌 Giriş
Bu proje kapsamında, New York City'deki Airbnb ilanlarının gecelik konaklama ücretlerini tahmin edebilen bir makine öğrenimi modeli geliştirilmiştir. Amacımız, hem ev sahiplerinin hem de kullanıcıların faydalanabileceği bir sistem önererek, piyasadaki dengesiz fiyatlandırmaların önüne geçmek ve veri temelli fiyat önerileri sunmaktır.

Proje, veri analizi, görselleştirme, model eğitimi ve değerlendirme aşamalarından oluşmakta olup; süreç boyunca hem teknik doğruluk hem de işlevsellik gözetilmiştir. Bu belge, projenin teknik detaylarını ayrıntılı bir biçimde ortaya koymaktadır.



## 🗂️ Veri Seti ve Problem Tanımı

Kullanılan veri seti, Airbnb'nin 2019 yılına ait New York City içindeki ilanlarını içermektedir. Her ilan için konum bilgileri, oda tipi, minimum konaklama süresi, kullanıcı yorumları gibi çeşitli nitelikler yer almaktadır.

- Bu veri seti ile çözülmek istenen temel problem şudur:  
**"Yeni bir Airbnb ilanı oluşturulduğunda, sahip olduğu özelliklere bakarak adil ve gerçekçi bir fiyat tahmini yapılabilir mi?"**


## 🧹 Veri Ön İşleme Adımları


Veri ön işleme, modelin başarısı üzerinde doğrudan etkili olan kritik bir aşamadır. Bu projede, New York City’deki Airbnb ilanlarını daha sağlıklı ve güvenilir bir şekilde analiz edebilmek adına aşağıdaki adımlar dikkatlice uygulanmıştır:

**1. 📦 Gereksiz Sütunların Kaldırılması**

Veri setinde model tahminine katkısı olmayan veya sadece kimlik bilgisi içeren sütunlar (`id`, `host_name`, `last_review`, `name`) veri setinden çıkarılmıştır. Bu tür sütunlar, modelin öğrenme sürecine anlamlı katkı sağlamadığı gibi, bazı durumlarda gürültü oluşturabilir. Ayrıca bu sütunların bazıları yüksek oranda benzersiz (örneğin `id`) olduğu için overfitting riskini artırır.

**2. 🧩 Eksik Verilerin Analizi ve Doldurulması**

`reviews_per_month` sütununda bazı satırlarda eksik değerler bulunmaktaydı. Bu eksik veriler, bu alanın `number_of_reviews` değeri sıfır olan satırlarda yer aldığını gösterdiği için, mantıksal olarak 0 ile doldurulmuştur. Bu sayede eksiklikten kaynaklı veri kaybı yaşanmadan işlemeye devam edilmiştir.

**3. 🚨 Aykırı Değerlerin Tespiti ve Temizlenmesi**

`price` ve `minimum_nights` gibi değişkenlerde uç değerlerin modelleme sürecini bozabileceği görülmüştür. Örneğin, bazı ilanlarda gecelik fiyatlar 10000 doların üzerindeyken, bazı ilanlarda minimum konaklama süresi 365 güne kadar çıkıyordu. Bu değerlerin, veri setinin genel dağılımından çok farklı olması, modelin genelleme yeteneğini zedeleyebilir. Bu nedenle:

- `price > 1000 olan ilanlar`,

- `minimum_nights > 30 olan ilanlar`

  kapsam dışı bırakılmış ve veri daha tutarlı hale getirilmiştir.

**4. 🔄 Sayısal Verilerin Dönüştürülmesi**

`reviews_per_month` değişkeni çarpık (skewed) bir dağılıma sahiptir. Doğrusal regresyon tabanlı bazı modeller bu tür verilerde zayıf performans gösterebilir. Bu nedenle **Yeo-Johnson** dönüşümü uygulanarak dağılım normalleştirilmiş ve modelin bu özelliği daha sağlıklı öğrenmesi sağlanmıştır.

|  Not: Yeo-Johnson dönüşümü, Box-Cox'un geliştirilmiş halidir ve sıfır ya da negatif değerler için de kullanılabilir.


![Ekran görüntüsü 2025-05-25 194137](https://github.com/user-attachments/assets/8de51f95-e8e1-4615-8ed1-8183ad163d42)


**5. 🏷️ Kategorik Verilerin Sayısallaştırılması**

`room_type`, `neighbourhood_group` gibi kategorik sütunlar, makine öğrenmesi modelleri tarafından doğrudan işlenemez. Bu nedenle bu değişkenler **One-Hot Encoding** yöntemiyle ikili sütunlara ayrılmıştır. 
Bu yöntem, kategoriler arasında yapay sıralama oluşturmadan her birini eşit uzaklıkta temsil eder, bu da özellikle ağaç tabanlı algoritmalar için avantaj sağlar.

**6. 📏 Özellik Ölçekleme (Model Bağımlı)**

Random Forest gibi ağaç tabanlı modeller ölçek duyarlı olmadıkları için (örneğin bir özelliğin 0-1 arası, diğerinin 0-1000 arasında olması sorun yaratmaz), tüm veri setinde standardizasyon uygulanmamıştır. Ancak regresyon gibi mesafe tabanlı modellerde kullanılması durumunda `MinMaxScaler` veya `StandardScaler` gibi yöntemlerin değerlendirilmesi gerektiği belirtilmiştir.

Bu ön işleme adımları sayesinde model hem daha sağlıklı öğrenme gerçekleştirmiş, hem de veri setindeki yapay çarpıklıklardan arındırılmış hale gelmiştir.

## 🔍 Keşifsel Veri Analizi (EDA)

Proje sürecine başlamadan önce veri setinin yapısını anlamak ve olası problemleri önceden tespit edebilmek adına kapsamlı bir EDA süreci yürütülmüştür. Bu analiz hem değişkenlerin doğasını ortaya koymak hem de modellerin başarısını artıracak çıkarımlar yapmak için kritik bir adımdır.

Yapılan başlıca analizler:

- **Oda Tipine Göre Ortalama Fiyatlar**: `room_type` değişkeninin fiyat üzerindeki etkisi analiz edilmiştir. Özellikle “Entire home/apt” kategorisinin ortalamanın çok üzerinde fiyatlara sahip olduğu görülmüştür.

![Ekran görüntüsü 2025-05-25 194206](https://github.com/user-attachments/assets/45ff4d91-3e28-43de-82e7-6c3a928ce0fe)

- **Bölgelere Göre Fiyat Dağılımları**: `neighbourhood_group` değişkeni kullanılarak her bölgedeki fiyat dağılımları incelenmiş ve Manhattan gibi merkezi lokasyonlarda fiyatların ciddi oranda yüksek olduğu gözlemlenmiştir.

- **Minimum Konaklama Süresi Dağılımı**: Özellikle çok yüksek değerler barındıran `minimum_nights` sütununda ciddi aykırılıklar (outlier) olduğu tespit edilerek bunların veri ön işleme adımlarında temizlenmesi gerektiğine karar verilmiştir.

- **Korelasyon Matrisi**: Sürekli değişkenler arasındaki korelasyonlar incelenmiş, yüksek korelasyon içeren değişkenler modelleme aşamasında dikkatle değerlendirilmiştir.

![Ekran görüntüsü 2025-05-25 194001](https://github.com/user-attachments/assets/35188f7e-95fb-4400-b87d-7095a8b5072d)


Bu adımda elde edilen bulgular sayesinde modellemede hangi değişkenlerin daha öncelikli olması gerektiği belirlenmiş, aynı zamanda veri kalitesini artırmak için gereken temizlik ve dönüşümler planlanmıştır.

---

## 🤖 Modelleme Süreci

Modelleme süreci, ilk olarak temel algoritmalarla denemeler yapılarak başlamıştır. Linear Regression, Decision Tree gibi modeller test edildikten sonra, en başarılı sonucu veren algoritma olarak Random Forest Regressor seçilmiştir.

Veri, `train_test_split` yöntemi ile %80 eğitim ve %20 test olacak şekilde ayrılmıştır. Model, `price` (fiyat) değişkenini hedef olarak alacak şekilde eğitilmiş ve tahminlerde bulunmuştur.

Model eğitim sürecinde şu adımlar izlenmiştir:

- Eğitim verisi ile model oluşturuldu

- Test verisi ile performans ölçüldü

- `r2_score` ve `mean_absolute_error` gibi metriklerle başarı değerlendirildi

- `feature_importances_` ile hangi özelliklerin daha belirleyici olduğu analiz edildi


Veri modeli, %80 eğitim ve %20 test olarak ikiye ayrıldı. `RandomForestRegressor` modeli kullanılarak tahminleme işlemi gerçekleştirildi.

![image](https://github.com/user-attachments/assets/61137f96-ad8b-4b04-a30a-f5734b804937)

## 📊 Model Performansı ve Metrikler-Lineer Regresyon Sonucu

| Metrik                  | Değer         |
|-------------------------|---------------|
| R² Skoru                | 0.54          |
| Mean Absolute Error     | 64.33         |
| Mean Squared Error      | 18117.5       |
| Root Mean Squared Error | 134.6         |

## 🌲 Neden Random Forest?

Random Forest, birçok karar ağacının bir araya gelerek oluşturduğu bir topluluk modelidir (ensemble learning). Bu modelin seçilmesinin başlıca nedenleri şunlardır:

- Overfitting'e karşı dirençlidir. Tek bir karar ağacı, veriye aşırı uyum gösterebilirken; Random Forest farklı örnekleme yöntemleriyle bu sorunu minimize eder.

- Özellik önemleri doğal olarak çıkarılabilir. Böylece hangi değişkenlerin model için daha açıklayıcı olduğunu analiz etmek kolaylaşır.

- Kategorik ve sayısal verilerle birlikte iyi çalışır. Özellikle karma veri tiplerinin olduğu durumlarda önemli bir avantaj sağlar.

- Eksik veriye duyarlılığı düşüktür. Ön işleme adımları iyi yapılmışsa, eksik ya da gürültülü veriler Random Forest'ın performansını fazla etkilemez.

Bu nedenlerle, proje kapsamındaki problem türü (regresyon) ve veri yapısı göz önüne alındığında en uygun ve başarılı model Random Forest olarak belirlenmiştir.

![Ekran görüntüsü 2025-05-25 194054](https://github.com/user-attachments/assets/fada4c10-6d41-4848-804b-398d96f7f35d)

## 📊 Model Performansı ve Metrikler- Random Forest Sonucu

| Metrik                  | Değer         |
|-------------------------|---------------|
| R² Skoru                | 0.99          |
| Mean Absolute Error     | 0.08          |
| Mean Squared Error      | 3.79          |


## 📌 Özellik Önem Değerlendirmesi

Modelin öğrenme sürecinde en etkili olan değişkenler analiz edilmiştir. Random Forest algoritması, her özelliğin model başarısına olan katkısını doğal olarak çıkarabildiği için bu analiz oldukça güvenilirdir.

Aşağıda bazı önemli değişkenlerin etkileri sıralanmıştır:


## 🗺️ Haritalar ile Coğrafi Görselleştirme

Airbnb ilanlarının konum bilgisini içermesi, projeye mekânsal bir boyut kazandırma fırsatı sunmuştur. Haritalar üzerinde yapılan görselleştirmeler, modelin anlaşılabilirliğini ve güvenilirliğini artırmak açısından oldukça değerlidir.

Harita analizinde yapılan başlıca işlemler:

- **Fiyatların Konuma Göre Dağılımı**: Her bir ilanın fiyatı, harita üzerinde `latitude` ve `longitude` koordinatları kullanılarak görselleştirilmiştir. Özellikle Manhattan civarında yüksek yoğunlukta pahalı ilanların bulunduğu gözlemlenmiştir.



- **Bölgelere Göre Renk Kodlamalı Görselleştirme**: `neighbourhood_group` kategorileri farklı renklerle işaretlenerek hangi bölgelerde yoğunluk olduğu ve fiyat farklarının mekânsal karşılıkları gösterilmiştir.

![Ekran görüntüsü 2025-05-25 193908](https://github.com/user-attachments/assets/97941be9-d5f2-499b-8280-86e3aa3a2f24)


- **Oda Tiplerinin Harita Üzerindeki Dağılımı**: `room_type` sütunu ile harita etkileşimli hale getirilmiş, Shared Room, Private Room, Entire Home gibi seçeneklerin nerelerde yoğunlaştığı gözlemlenmiştir.

  ![image](https://github.com/user-attachments/assets/a14817dd-06e7-4fc8-bc37-f762901a1b4a)


Bu bölümde oluşturulan haritalar, veri bilimi çıktılarının gerçek hayattaki karşılığını görselleştirerek projenin daha anlaşılır ve uygulanabilir hale gelmesini sağlamıştır. Özellikle yöneticilere, iş ortaklarına veya son kullanıcıya sunum yaparken bu tarz haritalar güçlü bir görsel destek sunar.

![Ekran görüntüsü 2025-05-25 193814n](https://github.com/user-attachments/assets/7bd663c2-4168-4ac3-88f6-62cf622de1f1)

![Ekran görüntüsü 2025-05-25 193732](https://github.com/user-attachments/assets/9647786b-761d-46ed-bdf6-22ee5a2a5b77)

## 📉 Tahmin Hatalarının Analizi

Modelin genel performansını değerlendirmek kadar, yaptığı hataları anlamak da oldukça önemlidir. Bu doğrultuda, test verisi üzerinde modelin tahmin ettiği fiyatlar ile gerçek fiyatlar karşılaştırılmış ve farklar incelenmiştir.

Yapılan hata analizi sonucunda, modelin özellikle uç değerlerde (örneğin çok lüks veya çok ucuz ilanlar) daha büyük sapmalar yaptığı gözlemlenmiştir. Bu durum, bazı özelliklerin (örneğin konumun tam adres seviyesi, daire büyüklüğü, iç tasarım gibi) veri setinde yer almamasından kaynaklanıyor olabilir.

Ayrıca hata dağılım grafikleri incelendiğinde, modelin düşük fiyat aralığında daha tutarlı tahminler yaptığı, ancak fiyat arttıkça sapmanın da arttığı anlaşılmıştır. Bu da regresyon modellerinde sık karşılaşılan bir durumdur ve logaritmik dönüşüm gibi yöntemlerle iyileştirmeye açık bir alandır.

![Ekran görüntüsü 2025-05-25 194023](https://github.com/user-attachments/assets/87755230-950e-491a-ad7d-f8495196a6bc)

![Ekran görüntüsü 2025-05-25 194036](https://github.com/user-attachments/assets/ee1138a8-2183-4b46-9406-eca99f4577d8)

Bu analiz, modelin güven aralığını belirlemek ve gelecekte yapılabilecek iyileştirme stratejileri için zemin hazırlamak açısından oldukça kıymetlidir.


## 🌍 Gerçek Hayattaki Uygulamalar

Geliştirilen bu model, gerçek dünyada birçok paydaş için doğrudan fayda sağlayabilecek niteliktedir. Öncelikle ev sahipleri, kendi ilanlarının konum, oda tipi ve müsaitlik gibi özelliklerine göre sistematik bir şekilde fiyat belirleyebilir. Bu, piyasa koşullarına göre fazla düşük ya da aşırı yüksek fiyat vermekten kaçınarak gelirlerini optimize etmelerine yardımcı olur. Aynı zamanda Airbnb kullanıcıları da benzer ilanlara göre fiyat kıyaslaması yaparak bütçelerine en uygun seçenekleri seçebilir; özellikle aynı mahallede benzer özelliklere sahip ilanlar arasındaki fiyat farklarını daha iyi analiz edebilirler. Airbnb gibi platform geliştiricileri ise bu modeli, ekstrem (anormal derecede düşük ya da yüksek) fiyatlı ilanları tespit etmek için bir filtreleme aracı olarak kullanabilir. Böylece sahte, spam veya fiyat manipülasyonu içeren ilanlar erken safhada tanımlanarak sistemden çıkarılabilir. Ek olarak, platforma yeni bölgeler eklendiğinde ya da farklı şehirlerde hizmet genişletildiğinde, bu model transfer learning yaklaşımlarıyla yeniden eğitilerek farklı lokasyonlara kolaylıkla adapte edilebilir. Pazarlama ekipleri ise model çıktılarından faydalanarak fiyat-performans açısından öne çıkan mahalleleri ve kullanıcıların yoğunlaştığı bölge türlerini tespit edebilir, böylece stratejik kampanyalar geliştirebilir.


## 🚀 Gelecek Geliştirmeler ve Öneriler

Proje, hali hazırda sağlam bir temel üzerine kurulmuş olsa da, gelecekte şu yönlerde geliştirilebilir:

- **Zaman Serisi Analizi:** Fiyatlar yılın farklı dönemlerinde değişiklik gösteriyor olabilir. Tarihsel trendlere göre fiyat tahmini için zaman serisi analizleri entegre edilebilir.

- **Derin Öğrenme Modelleri:** Çok katmanlı yapay sinir ağları veya LSTM gibi modellerle daha kompleks örüntüler yakalanabilir.

- **Arayüz Geliştirme (UI):** Streamlit ya da Dash gibi kütüphanelerle, kullanıcıların giriş yaparak tahmini fiyat alabilecekleri bir uygulama geliştirilebilir.

- **Gerçek Zamanlı Veri Kullanımı:** API ile gerçek zamanlı veri çekilerek dinamik fiyat tahmini yapılabilir.

- **Anomali Tespiti:** Aykırı fiyatları otomatik tespit eden bir modül eklenebilir.

## 🔗 İlgili Bağlantılar

- [📁 Kaggle Notebook 1](https://www.kaggle.com/code/ilaydaakarahan/airbnb-price-estimate)
- [📁 Kaggle Notebook 2](https://www.kaggle.com/code/ilaydaakarahan/airbnb-price-estimate?select=AB_NYC_2019.csv)
