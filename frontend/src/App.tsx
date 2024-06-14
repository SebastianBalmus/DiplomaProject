import { useState } from 'react';
import axios from 'axios';
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline';
import AppBar from './components/AppBar';
import TTSForm from './components/TTSForm';
import Footer from './components/Footer';
import darkTheme from './theme'
import LoadingScreeen from './components/LoadingScreen';
import Player from './components/Player';


function App() {
  const [wavUrl, setWavUrl] = useState<string>('');
  const [melUrl, setMelUrl] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const onFormSubmit = async (values: {text: string, model: string}) => {
    setLoading(true)
    const { data } = await axios.post(
      'http://46.243.115.3:8080/infer',
      {
        text: values.text,
      },
      {
        params: {
          model_id: values.model,
          use_cuda: true,
        },
        headers: {
          Accept: 'audio/wav',
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
        responseType: 'blob',
      },
    );

    const url = URL.createObjectURL(data);
    setWavUrl(url)

    const formData = new FormData();
    formData.append('wav_file', data, 'audio.wav');

    const melResponse = await axios.post('http://46.243.115.3:8080/mel', 
      formData,
      {
      headers: {
        'Content-Type': 'audio/wav',
        'Access-Control-Allow-Origin': '*',
      },
      responseType: 'arraybuffer',
    });

    const melUrl = URL.createObjectURL(new Blob([melResponse.data], { type: 'image/png' }));
    setMelUrl(melUrl);

    setLoading(false)
  }

  const resetAction = () => {
    setWavUrl('');
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <AppBar />
      {loading ? (
        <LoadingScreeen />
      ) : wavUrl !== "" ? (
        <Player
          wav_src={wavUrl}
          mel_src={melUrl}
          resetAction={resetAction}
        />
      ) : (
        <TTSForm onSubmit={onFormSubmit} />
      )}
      <Footer />
    </ThemeProvider>
  )
}

export default App;
