import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import spg_logo from '../assets/logo_spg_white_small.png';
import etti_logo from '../assets/sigla_ETTI.png';


function Footer () {
  return (
    <AppBar
      position="static"
      color="primary"
      sx={{
        top: 'auto',
        bottom: 0,
        width: '100%',
        padding: '10px 0',
        textAlign: 'center',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '20px',
          gap: '20px'
        }}
      >
        <a href="https://etti.utcluj.ro/">
          <img src={etti_logo} height='80px' />
        </a>
        <a href="https://speech.utcluj.ro/">
          <img src={spg_logo} height='80px' />
        </a>
      </Box>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Typography
            variant="h6"
            noWrap
            sx={{
              display: 'flex',
            }}
          >
            © Balmuș Sebastian 2024
          </Typography>
      </Box>
    </AppBar>
  )
}

export default Footer;
