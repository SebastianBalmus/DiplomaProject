import { useMediaQuery } from '@mui/material';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import AudioPlayer from 'react-h5-audio-player';
import 'react-h5-audio-player/lib/styles.css';
import darkTheme from '../theme'
import './Player.css'


function Player(props: {src: string, resetAction: () => void}) {
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
      <AudioPlayer
        src={props.src}
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
