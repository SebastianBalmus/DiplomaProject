import { useMediaQuery } from '@mui/material';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import AudioPlayer from 'react-h5-audio-player';
import 'react-h5-audio-player/lib/styles.css';
import darkTheme from '../theme'
import './Player.css'


function Player(props: {wav_src: string, mel_src: string, resetAction: () => void}) {
  const isDesktop = useMediaQuery(darkTheme.breakpoints.up('md'));

  return (
    <Box
    sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      width: '100vw',
      height: '100vh',
    }}
  >
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        width: isDesktop ? '50%' : '100%',
        height: '100%',
        padding: '40px',
        gap: '20px'
      }}
    >
      <img
        src={props.mel_src}
        style={{
          maxWidth: '100%',
        }}
      />
      <AudioPlayer
        src={props.wav_src}
        customAdditionalControls={[]}
        showJumpControls={false}
      />
      <Button
        fullWidth
        size="large"
        variant="contained"
        onClick={props.resetAction}
        sx={{
          backgroundColor: 'primary.dark'
        }}
      >
        Try another sentence
      </Button>
    </Box>
  </Box>
  )
}

export default Player;
