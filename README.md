# Anomalous sound detection based on differential features of multi channel acoustic signals considering spatial and temporal variations

[Paper](https://www.ieice.org/publications/ken/summary.php?contribution_id=123948&society_cd=ISS&ken_id=IE&year=2023&presen_date=2023-03-01&schedule_id=7826&lang=en&expandable=3): Anomalous sound detection based on differential features of multi channel acoustic signals considering spatial and temporal variations  
Author: Shota Nishiyama, Akira Tamamori  
Affiliation, Organization: Aichi Institute of Technology Graduate School of Business Administration and Computer Science (AIT)  
keyword: Anomalous sound detection, multi-channel-signal, differential features

## Abstract

Anomalous sound detection plays an essential role in machine condition management in factory automation. The task of anomalous sound detection is to distinguish between normal and anomalous sounds. Collecting anomalous sounds in advance is difficult because they occur infrequently and are very diverse. Therefore, anomalous sound detection is treated as a problem of only detecting anomalous sounds from normal sounds. Most anomalous sound detection methods target single-channel audio signals. On the other hand, in some factories, multiple microphones can be installed to record multi-channel audio signals. In this study, we propose a feature set that is useful for the anomalous sound detection task, targeting multi-channel audio signals obtained from multiple microphones at different distances from the sound source. In addition to the phase and mel-spectrogram obtained from the complex-spectrogram, their differential features are used as input to the model to account for changes in time and space due to the distance between the microphones and the sound sources. The dataset is a multi-channel audio signal from ToyADMOS. Comparison experiments show that the proposed features significantly improve the AUC, an evaluation index, compared to the accuracy of anomalous sound detection using only the mel-spectrogram as a feature, indicating the usefulness of differential features that take into account temporal and spatial variations.

## Usege

```bash
cd environments/gpu
docker compose up -d
docker compose exec real-toyadmos bash
poetry install
poetry run bash all_features_run.sh
```

> [!NOTE]  
plase rewrite your environment -> `configures/experiments/default.yaml`

## Citation

```
@techreport{weko_224457_1,
   author	 = "西山,翔大 and 玉森,聡",
   title	 = "マルチチャネル音声信号の時間的・空間的な変化を考慮した微分特徴量に基づく異常音検知手法",
   year 	 = "2023",
   institution	 = "愛知工大, 愛知工大",
   number	 = "60",
   month	 = "feb"
}
```