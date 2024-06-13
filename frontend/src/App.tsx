import axios from 'axios';
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline';
import AppBar from './components/AppBar';
import TTSForm from './components/TTSForm';
import Footer from './components/Footer';
import darkTheme from './theme'

const onFormSubmit = async (values: {text: string, model: string}) => {
  const { data } = await axios.post(
    'http://46.243.115.3:8080/infer',
    {
      text: values.text,
    },
    {
      params: {
        model: values.model,
        use_cuda: true,
      },
      headers: {
        Accept: 'audio/wav',
        'Content-Type': 'audio/wav',
        'Access-Control-Allow-Origin': '*',
      }
    },
  );

  console.log(data)
  return data
}

function App() {


  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <AppBar />
      <TTSForm onSubmit={onFormSubmit} />
      <Footer />
    </ThemeProvider>
  )
}

export default App;
