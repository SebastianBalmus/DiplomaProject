import Box from '@mui/material/Box';
import { ScaleLoader } from 'react-spinners';


function LoadingScreeen() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100vw',
        height: '100vh',
        padding: '40px',
        gap: '20px'
      }}
    >
      <ScaleLoader
        color="#55A6F6"
        height={90}
        width={8}
        radius={20}
        margin={6}
      />
    </Box>
  );
}

export default LoadingScreeen;
