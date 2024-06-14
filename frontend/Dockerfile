# Serve the React application using Nginx
FROM nginx:alpine

# Copy the built React files from the build stage
COPY dist /usr/share/nginx/html

# Expose the port the app runs on
EXPOSE 80

# Command to run the nginx server
CMD ["nginx", "-g", "daemon off;"]
